#include "managers.h"

#include <netcdf.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>
#include <regex>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

int open_height_variable(const std::string& filename,
    int& ncid,
    int& varid,
    std::size_t (&dims)[3])
{
    int retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
    if (retval != NC_NOERR) {
        std::cerr << "Failed to open file " << filename << " : "
                  << nc_strerror(retval) << std::endl;
        return retval;
    }

    retval = nc_inq_varid(ncid, "height", &varid);
    if (retval != NC_NOERR) {
        std::cerr << "Variable 'height' is missing in " << filename << std::endl;
        nc_close(ncid);
        ncid = -1;
        return retval;
    }

    int ndims = 0;
    retval = nc_inq_varndims(ncid, varid, &ndims);
    if (retval != NC_NOERR || ndims != 3) {
        std::cerr << "Expected variable 'height' to have 3 dimensions in "
                  << filename << std::endl;
        nc_close(ncid);
        ncid = -1;
        return retval == NC_NOERR ? NC_EINVAL : retval;
    }

    int dimids[3] = { 0, 0, 0 };
    retval = nc_inq_vardimid(ncid, varid, dimids);
    if (retval != NC_NOERR) {
        std::cerr << "Failed to query dimension identifiers in " << filename
                  << " : " << nc_strerror(retval) << std::endl;
        nc_close(ncid);
        ncid = -1;
        return retval;
    }

    for (int i = 0; i < 3; ++i) {
        retval = nc_inq_dimlen(ncid, dimids[i], &dims[i]);
        if (retval != NC_NOERR) {
            std::cerr << "Failed to query dimension length in " << filename
                      << " : " << nc_strerror(retval) << std::endl;
            nc_close(ncid);
            ncid = -1;
            return retval;
        }
    }

    return NC_NOERR;
}

void close_netcdf(int& ncid)
{
    if (ncid >= 0) {
        nc_close(ncid);
        ncid = -1;
    }
}

int extract_index(const fs::path& file_path)
{
    static const std::regex pattern("_(\\d+)\\.nc");
    std::smatch match;
    std::string filename = file_path.filename().string();
    if (std::regex_search(filename, match, pattern)) {
        return std::stoi(match[1].str());
    }
    return std::numeric_limits<int>::max();
}

std::vector<fs::path> get_sorted_nc_files(const std::string& folder)
{
    std::vector<fs::path> files;
    if (!fs::exists(folder)) {
        return files;
    }

    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto& path = entry.path();
        if (path.extension() == ".nc"
            && path.filename().string().find('_') != std::string::npos) {
            files.push_back(path);
        }
    }

    std::sort(files.begin(), files.end(), [](const fs::path& lhs, const fs::path& rhs) {
        return extract_index(lhs) < extract_index(rhs);
    });

    return files;
}

} // namespace

WaveManager::WaveManager(const std::string& nc_file_)
    : nc_file(nc_file_)
{
    if (open_height_variable(nc_file, ncid, varid, dims) != NC_NOERR) {
        close_netcdf(ncid);
        varid = -1;
        dims[0] = dims[1] = dims[2] = 0;
    }
}

WaveManager::~WaveManager()
{
    close_netcdf(ncid);
}

WaveManager::WaveManager(WaveManager&& other) noexcept
    : ncid(other.ncid)
    , varid(other.varid)
    , dims{ other.dims[0], other.dims[1], other.dims[2] }
    , nc_file(std::move(other.nc_file))
{
    other.ncid = -1;
    other.varid = -1;
    other.dims[0] = other.dims[1] = other.dims[2] = 0;
}

WaveManager& WaveManager::operator=(WaveManager&& other) noexcept
{
    if (this != &other) {
        close_netcdf(ncid);
        ncid = other.ncid;
        varid = other.varid;
        dims[0] = other.dims[0];
        dims[1] = other.dims[1];
        dims[2] = other.dims[2];
        nc_file = std::move(other.nc_file);

        other.ncid = -1;
        other.varid = -1;
        other.dims[0] = other.dims[1] = other.dims[2] = 0;
    }
    return *this;
}

bool WaveManager::valid() const noexcept
{
    return ncid >= 0 && varid >= 0;
}

bool WaveManager::get_dimensions(std::size_t& T, std::size_t& Y, std::size_t& X) const
{
    if (!valid()) {
        return false;
    }
    T = dims[0];
    Y = dims[1];
    X = dims[2];
    return true;
}

std::vector<std::vector<std::vector<double>>> WaveManager::load_block(
    int y_start,
    int y_count,
    int x_start,
    int x_count) const
{
    std::vector<std::vector<std::vector<double>>> data;
    if (!valid() || y_count <= 0 || x_count <= 0) {
        return data;
    }

    if (y_start < 0 || x_start < 0) {
        return data;
    }

    std::size_t y_end = static_cast<std::size_t>(y_start) + static_cast<std::size_t>(y_count);
    std::size_t x_end = static_cast<std::size_t>(x_start) + static_cast<std::size_t>(x_count);
    if (y_end > dims[1] || x_end > dims[2]) {
        return data;
    }

    std::size_t T = dims[0];
    data.assign(T, std::vector<std::vector<double>>(static_cast<std::size_t>(y_count),
        std::vector<double>(static_cast<std::size_t>(x_count), 0.0)));

    std::vector<double> buffer(T * static_cast<std::size_t>(y_count) * static_cast<std::size_t>(x_count));
    size_t start[3] = { 0, static_cast<size_t>(y_start), static_cast<size_t>(x_start) };
    size_t count[3] = { T, static_cast<size_t>(y_count), static_cast<size_t>(x_count) };

    std::cout << "Loading block y=[" << y_start << ", " << (y_start + y_count)
              << ") x=[" << x_start << ", " << (x_start + x_count) << ") across "
              << T << " time steps from " << nc_file << std::endl;

    int retval = nc_get_vara_double(ncid, varid, start, count, buffer.data());
    if (retval != NC_NOERR) {
        std::cerr << "Failed to read variable data from " << nc_file << " : "
                  << nc_strerror(retval) << std::endl;
        return {};
    }

    std::cout << "Completed block load from " << nc_file << std::endl;

    for (std::size_t t = 0; t < T; ++t) {
        for (int y = 0; y < y_count; ++y) {
            for (int x = 0; x < x_count; ++x) {
                std::size_t index = (t * static_cast<std::size_t>(y_count) + static_cast<std::size_t>(y))
                    * static_cast<std::size_t>(x_count) + static_cast<std::size_t>(x);
                data[t][static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] = buffer[index];
            }
        }
    }

    return data;
}

BasisManager::BasisManager(const std::string& folder_)
    : folder(folder_)
{
    auto files = get_sorted_nc_files(folder);
    bases.reserve(files.size());
    for (const auto& file : files) {
        BasisEntry entry;
        entry.path = file.string();
        if (open_height_variable(entry.path, entry.ncid, entry.varid, entry.dims) == NC_NOERR) {
            bases.push_back(std::move(entry));
        }
    }
}

BasisManager::~BasisManager()
{
    for (auto& basis : bases) {
        close_netcdf(basis.ncid);
    }
}

BasisManager::BasisManager(BasisManager&& other) noexcept
    : bases(std::move(other.bases))
    , folder(std::move(other.folder))
{
    other.bases.clear();
}

BasisManager& BasisManager::operator=(BasisManager&& other) noexcept
{
    if (this != &other) {
        for (auto& basis : bases) {
            close_netcdf(basis.ncid);
        }
        bases = std::move(other.bases);
        folder = std::move(other.folder);
        other.bases.clear();
    }
    return *this;
}

bool BasisManager::valid() const noexcept
{
    return !bases.empty();
}

std::size_t BasisManager::basis_count() const
{
    return bases.size();
}

bool BasisManager::get_dimensions(std::size_t& T, std::size_t& Y, std::size_t& X) const
{
    if (!valid()) {
        return false;
    }
    const auto& dims_ref = bases.front().dims;
    T = dims_ref[0];
    Y = dims_ref[1];
    X = dims_ref[2];
    return true;
}

std::vector<std::vector<std::vector<std::vector<double>>>> BasisManager::load_block(
    int y_start,
    int y_count,
    int x_start,
    int x_count) const
{
    std::vector<std::vector<std::vector<std::vector<double>>>> data;
    if (!valid() || y_count <= 0 || x_count <= 0) {
        return data;
    }

    if (y_start < 0 || x_start < 0) {
        return data;
    }

    std::size_t y_end = static_cast<std::size_t>(y_start) + static_cast<std::size_t>(y_count);
    std::size_t x_end = static_cast<std::size_t>(x_start) + static_cast<std::size_t>(x_count);
    if (y_end > bases.front().dims[1] || x_end > bases.front().dims[2]) {
        return data;
    }

    std::size_t T = bases.front().dims[0];
    data.assign(bases.size(), std::vector<std::vector<std::vector<double>>>(T,
        std::vector<std::vector<double>>(static_cast<std::size_t>(y_count),
            std::vector<double>(static_cast<std::size_t>(x_count), 0.0))));

    std::vector<double> buffer(T * static_cast<std::size_t>(y_count) * static_cast<std::size_t>(x_count));

    for (std::size_t basis_idx = 0; basis_idx < bases.size(); ++basis_idx) {
        const auto& basis = bases[basis_idx];
        size_t start[3] = { 0, static_cast<size_t>(y_start), static_cast<size_t>(x_start) };
        size_t count[3] = { T, static_cast<size_t>(y_count), static_cast<size_t>(x_count) };

        std::cout << "Loading block y=[" << y_start << ", " << (y_start + y_count)
                  << ") x=[" << x_start << ", " << (x_start + x_count) << ") across "
                  << T << " time steps from " << basis.path << std::endl;

        int retval = nc_get_vara_double(basis.ncid, basis.varid, start, count, buffer.data());
        if (retval != NC_NOERR) {
            std::cerr << "Failed to read variable data from " << basis.path
                      << " : " << nc_strerror(retval) << std::endl;
            return {};
        }

        std::cout << "Completed block load from " << basis.path << std::endl;

        for (std::size_t t = 0; t < T; ++t) {
            for (int y = 0; y < y_count; ++y) {
                for (int x = 0; x < x_count; ++x) {
                    std::size_t index = (t * static_cast<std::size_t>(y_count) + static_cast<std::size_t>(y))
                        * static_cast<std::size_t>(x_count) + static_cast<std::size_t>(x);
                    data[basis_idx][t][static_cast<std::size_t>(y)][static_cast<std::size_t>(x)] = buffer[index];
                }
            }
        }
    }

    return data;
}
