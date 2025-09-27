#include "managers.h"
#include <netcdf.h>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <future>
#include <regex>
#include <chrono>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace {

constexpr const char* kHeightVariable = "height";

int open_nc_file(const std::string& filename, int& ncid) {
    int retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
    if (retval != NC_NOERR) {
        std::cerr << "[ERROR] Failed to open file " << filename << ": " << nc_strerror(retval) << std::endl;
    }
    return retval;
}

bool read_nc_dimensions(const fs::path& file, const char* variable_name, size_t& T, size_t& Y, size_t& X) {
    int ncid;
    if (open_nc_file(file.string(), ncid) != NC_NOERR) {
        return false;
    }

    int varid;
    int retval = nc_inq_varid(ncid, variable_name, &varid);
    if (retval != NC_NOERR) {
        std::cerr << "[ERROR] Variable '" << variable_name << "' not found in " << file.string() << std::endl;
        nc_close(ncid);
        return false;
    }

    int ndims = 0;
    nc_inq_varndims(ncid, varid, &ndims);
    if (ndims != 3) {
        std::cerr << "[ERROR] Expected 3 dimensions in " << file.string() << std::endl;
        nc_close(ncid);
        return false;
    }

    int dimids[3];
    nc_inq_vardimid(ncid, varid, dimids);
    nc_inq_dimlen(ncid, dimids[0], &T);
    nc_inq_dimlen(ncid, dimids[1], &Y);
    nc_inq_dimlen(ncid, dimids[2], &X);
    nc_close(ncid);
    return true;
}

std::vector<std::vector<std::vector<double>>> read_nc_file(const fs::path& file, int y_start, int y_end) {
    const auto load_begin = Clock::now();

    std::vector<std::vector<std::vector<double>>> data;
    int ncid;

    if (open_nc_file(file.string(), ncid) != NC_NOERR) {
        return data;
    }

    int varid;
    int retval = nc_inq_varid(ncid, kHeightVariable, &varid);
    if (retval != NC_NOERR) {
        std::cerr << "[ERROR] Variable '" << kHeightVariable << "' not found in " << file.string() << std::endl;
        nc_close(ncid);
        return data;
    }

    int ndims = 0;
    nc_inq_varndims(ncid, varid, &ndims);
    if (ndims != 3) {
        std::cerr << "[ERROR] Expected 3 dimensions in " << file.string() << std::endl;
        nc_close(ncid);
        return data;
    }

    int dimids[3];
    size_t T = 0, Y = 0, X = 0;
    nc_inq_vardimid(ncid, varid, dimids);
    nc_inq_dimlen(ncid, dimids[0], &T);
    nc_inq_dimlen(ncid, dimids[1], &Y);
    nc_inq_dimlen(ncid, dimids[2], &X);

    int local_y_end = std::min<int>(y_end, static_cast<int>(Y));
    if (local_y_end <= y_start) {
        nc_close(ncid);
        return data;
    }

    const size_t region_height = static_cast<size_t>(local_y_end - y_start);
    data.resize(T, std::vector<std::vector<double>>(region_height, std::vector<double>(X, 0.0)));

    size_t start[3] = { 0, static_cast<size_t>(y_start), 0 };
    size_t count[3] = { T, region_height, X };

    std::vector<double> buffer(T * region_height * X, 0.0);
    retval = nc_get_vara_double(ncid, varid, start, count, buffer.data());
    if (retval != NC_NOERR) {
        std::cerr << "[ERROR] Failed to read file " << file.string() << ": " << nc_strerror(retval) << std::endl;
        nc_close(ncid);
        return data;
    }
    nc_close(ncid);

    for (size_t t = 0; t < T; ++t) {
        for (size_t i = 0; i < region_height; ++i) {
            for (size_t x = 0; x < X; ++x) {
                const size_t idx = t * region_height * X + i * X + x;
                data[t][i][x] = buffer[idx];
            }
        }
    }

    const auto load_end = Clock::now();
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_begin).count();
    std::cout << "[TIMING] Loaded " << file << " in " << elapsed_ms << " ms" << std::endl;

    return data;
}

int extractIndex(const fs::path& filePath) {
    std::regex regexPattern("_(\\d+)\\.nc");
    std::smatch match;
    const std::string filename = filePath.filename().string();
    if (std::regex_search(filename, match, regexPattern)) {
        return std::stoi(match[1].str());
    }
    return std::numeric_limits<int>::max();
}

std::vector<fs::path> getSortedFileList(const std::string& folder) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) {
            continue;
        }
        const auto filePath = entry.path();
        if (filePath.extension() == ".nc" && filePath.filename().string().find('_') != std::string::npos) {
            files.push_back(filePath);
        }
    }

    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return extractIndex(a) < extractIndex(b);
    });

    return files;
}

} // namespace

std::vector<std::vector<std::vector<double>>> WaveManager::load_mariogramm_by_region(int y_start, int y_end) {
    return read_nc_file(nc_file, y_start, y_end);
}

std::vector<std::vector<std::vector<std::vector<double>>>> BasisManager::get_fk_region(int y_start, int y_end) {
    std::vector<std::vector<std::vector<std::vector<double>>>> fk;
    auto files = getSortedFileList(folder);
    for (const auto& file : files) {
        auto file_data = read_nc_file(file, y_start, y_end);
        if (!file_data.empty()) {
            fk.push_back(std::move(file_data));
        }
    }
    return fk;
}

bool WaveManager::describe(size_t& time_dim, size_t& y_dim, size_t& x_dim) const {
    return read_nc_dimensions(nc_file, kHeightVariable, time_dim, y_dim, x_dim);
}

bool BasisManager::describe(size_t& dataset_count, size_t& time_dim, size_t& y_dim, size_t& x_dim) const {
    const auto files = getSortedFileList(folder);
    if (files.empty()) {
        return false;
    }
    if (!read_nc_dimensions(files.front(), kHeightVariable, time_dim, y_dim, x_dim)) {
        return false;
    }
    dataset_count = files.size();
    return true;
}

