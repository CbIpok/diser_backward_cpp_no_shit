#include "managers.h"

#include <netcdf.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <regex>
#include <system_error>
#include <mutex>

namespace {

constexpr const char* kVariableName = "height";

std::mutex& netcdf_mutex() {
    static std::mutex mutex;
    return mutex;
}

struct NCFileHandle {
    int id = -1;

    explicit NCFileHandle(const std::string& path) {
        if (nc_open(path.c_str(), NC_NOWRITE, &id) != NC_NOERR) {
            id = -1;
        }
    }

    ~NCFileHandle() {
        if (id >= 0) {
            nc_close(id);
        }
    }

    bool valid() const { return id >= 0; }
};

struct ContiguousSpan {
    int y = 0;
    int x_start = 0;
    int length = 0;
    std::size_t point_offset = 0;
};

bool inquire_variable_info(int ncid, NetCDFVariableInfo& info) {
    int varid = -1;
    if (nc_inq_varid(ncid, kVariableName, &varid) != NC_NOERR) {
        std::cerr << "Failed to locate variable '" << kVariableName << "' in NetCDF file" << std::endl;
        return false;
    }

    int ndims = 0;
    if (nc_inq_varndims(ncid, varid, &ndims) != NC_NOERR || ndims != 3) {
        std::cerr << "Unexpected variable dimensionality for '" << kVariableName << "'" << std::endl;
        return false;
    }

    int dimids[3];
    if (nc_inq_vardimid(ncid, varid, dimids) != NC_NOERR) {
        std::cerr << "Failed to inquire variable dimensions" << std::endl;
        return false;
    }

    size_t lengths[3];
    for (int i = 0; i < 3; ++i) {
        if (nc_inq_dimlen(ncid, dimids[i], &lengths[i]) != NC_NOERR) {
            std::cerr << "Failed to read dimension length index " << i << std::endl;
            return false;
        }
    }

    info.t = lengths[0];
    info.y = lengths[1];
    info.x = lengths[2];
    return true;
}

std::vector<ContiguousSpan> build_spans(const Point2i* points, std::size_t count) {
    std::vector<ContiguousSpan> spans;
    spans.reserve(count);

    if (count == 0) {
        return spans;
    }

    ContiguousSpan current;
    current.y = bg::get<1>(points[0]);
    current.x_start = bg::get<0>(points[0]);
    current.length = 1;
    current.point_offset = 0;
    int prev_x = current.x_start;

    for (std::size_t i = 1; i < count; ++i) {
        int x = bg::get<0>(points[i]);
        int y = bg::get<1>(points[i]);
        if (y == current.y && x == prev_x + 1) {
            ++current.length;
            prev_x = x;
            continue;
        }
        spans.push_back(current);
        current.y = y;
        current.x_start = x;
        current.length = 1;
        current.point_offset = i;
        prev_x = x;
    }
    spans.push_back(current);

    return spans;
}

bool read_points_from_file(const std::string& file,
    const std::vector<ContiguousSpan>& spans,
    const NetCDFVariableInfo& info,
    std::vector<double>& destination) {

    if (spans.empty()) {
        return true;
    }

    std::lock_guard<std::mutex> lock(netcdf_mutex());

    NCFileHandle handle(file);
    if (!handle.valid()) {
        std::cerr << "Failed to open NetCDF file: " << file << std::endl;
        return false;
    }

    int varid = -1;
    if (nc_inq_varid(handle.id, kVariableName, &varid) != NC_NOERR) {
        std::cerr << "Failed to locate variable '" << kVariableName << "' in NetCDF file" << std::endl;
        return false;
    }

    std::size_t max_length = 0;
    for (const auto& span : spans) {
        max_length = std::max(max_length, static_cast<std::size_t>(span.length));
    }

    std::vector<double> buffer(info.t * max_length);

    for (const auto& span : spans) {
        if (span.length <= 0) {
            continue;
        }
        if (span.y < 0 || span.x_start < 0) {
            std::cerr << "Encountered negative coordinates during NetCDF read" << std::endl;
            return false;
        }
        if (static_cast<std::size_t>(span.y) >= info.y) {
            std::cerr << "Y coordinate out of bounds during NetCDF read" << std::endl;
            return false;
        }
        if (static_cast<std::size_t>(span.x_start + span.length) > info.x) {
            std::cerr << "X range out of bounds during NetCDF read" << std::endl;
            return false;
        }

        size_t start[3] = {
            0,
            static_cast<size_t>(span.y),
            static_cast<size_t>(span.x_start)
        };
        size_t count[3] = {
            info.t,
            1,
            static_cast<size_t>(span.length)
        };

        std::size_t values_to_copy = info.t * static_cast<std::size_t>(span.length);
        int retval = nc_get_vara_double(handle.id, varid, start, count, buffer.data());
        if (retval != NC_NOERR) {
            std::cerr << "Failed to read data from NetCDF file: " << nc_strerror(retval) << std::endl;
            return false;
        }

        for (std::size_t t = 0; t < info.t; ++t) {
            for (int dx = 0; dx < span.length; ++dx) {
                std::size_t source_index = t * static_cast<std::size_t>(span.length) + static_cast<std::size_t>(dx);
                std::size_t point_index = span.point_offset + static_cast<std::size_t>(dx);
                std::size_t dest_index = point_index * info.t + t;
                if (source_index >= values_to_copy || dest_index >= destination.size()) {
                    std::cerr << "Indexing error while copying NetCDF data" << std::endl;
                    return false;
                }
                destination[dest_index] = buffer[source_index];
            }
        }
    }

    return true;
}

int extract_index(const std::filesystem::path& path) {
    static const std::regex pattern("_(\\d+)\\.nc");
    std::smatch match;
    const std::string name = path.filename().string();
    if (std::regex_search(name, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return std::numeric_limits<int>::max();
}

std::vector<std::filesystem::path> get_sorted_file_list(const std::string& folder) {
    std::vector<std::filesystem::path> files;
    std::error_code ec;
    std::filesystem::directory_iterator it(folder, ec);
    if (ec) {
        std::cerr << "Failed to iterate folder " << folder << ": " << ec.message() << std::endl;
        return files;
    }

    for (; it != std::filesystem::directory_iterator(); it.increment(ec)) {
        if (ec) {
            std::cerr << "Error while iterating folder " << folder << ": " << ec.message() << std::endl;
            break;
        }
        const auto& entry = *it;
        if (!entry.is_regular_file()) {
            continue;
        }
        auto path = entry.path();
        if (path.extension() != ".nc") {
            continue;
        }
        files.push_back(path);
    }

    std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) {
        int ia = extract_index(a);
        int ib = extract_index(b);
        if (ia != ib) {
            return ia < ib;
        }
        return a < b;
    });

    return files;
}

} // namespace

BasisManager::BasisManager(const std::string& folder_) : folder(folder_) {
    basis_files = get_sorted_file_list(folder);
    if (!basis_files.empty()) {
        NCFileHandle handle(basis_files.front().string());
        if (handle.valid() && inquire_variable_info(handle.id, info)) {
            has_info = true;
        }
        else {
            std::cerr << "Failed to read basis NetCDF metadata from " << basis_files.front() << std::endl;
        }
    }
}

int BasisManager::basis_count() const {
    return static_cast<int>(basis_files.size());
}

const NetCDFVariableInfo& BasisManager::describe() const {
    return info;
}

bool BasisManager::load_points(const Point2i* points, std::size_t count, std::vector<std::vector<double>>& out) const {
    if (count == 0) {
        out.clear();
        return true;
    }
    if (!has_info) {
        std::cerr << "Basis metadata is unavailable" << std::endl;
        return false;
    }
    if (basis_files.empty()) {
        std::cerr << "No basis NetCDF files found" << std::endl;
        return false;
    }

    auto spans = build_spans(points, count);
    out.assign(basis_files.size(), std::vector<double>(count * info.t, 0.0));

    for (std::size_t idx = 0; idx < basis_files.size(); ++idx) {
        if (!read_points_from_file(basis_files[idx].string(), spans, info, out[idx])) {
            return false;
        }
    }

    return true;
}

WaveManager::WaveManager(const std::string& nc_file_) : nc_file(nc_file_) {
    NCFileHandle handle(nc_file);
    if (handle.valid() && inquire_variable_info(handle.id, info)) {
        has_info = true;
    }
    else {
        std::cerr << "Failed to read wave NetCDF metadata from " << nc_file << std::endl;
    }
}

const NetCDFVariableInfo& WaveManager::describe() const {
    return info;
}

bool WaveManager::valid() const {
    return has_info;
}

bool WaveManager::load_points(const Point2i* points, std::size_t count, std::vector<double>& out) const {
    out.clear();
    if (count == 0) {
        return true;
    }
    if (!has_info) {
        std::cerr << "Wave metadata is unavailable" << std::endl;
        return false;
    }

    auto spans = build_spans(points, count);
    out.assign(count * info.t, 0.0);
    return read_points_from_file(nc_file, spans, info, out);
}
