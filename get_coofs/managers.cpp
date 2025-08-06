#include "managers.h"
#include <netcdf.h>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <vector>
#include <future>
#include <regex>
namespace fs = std::filesystem;

// ������� ��� �������� NetCDF-����� � ��������� ������
int open_nc_file(const std::string& filename, int& ncid) {
    int retval = nc_open(filename.c_str(), NC_NOWRITE, &ncid);
    if (retval != NC_NOERR) {
        std::cerr << "������ �������� ����� " << filename << " : " << nc_strerror(retval) << std::endl;
    }
    return retval;
}
std::vector<std::vector<std::vector<double>>> read_nc_file(const fs::path& file, int y_start, int y_end) {
    std::vector<std::vector<std::vector<double>>> data;
    std::cout << "loading: " << file << std::endl;
    int ncid;

    if (open_nc_file(file.string(), ncid) != NC_NOERR)
        return data;

    int varid;
    int retval = nc_inq_varid(ncid, "height", &varid);
    if (retval != NC_NOERR) {
        std::cerr << "���������� 'height' �� ������� � " << file.string() << std::endl;
        nc_close(ncid);
        return data;
    }

    int ndims;
    nc_inq_varndims(ncid, varid, &ndims);
    if (ndims != 3) {
        std::cerr << "��������� 3 ��������� � ����� " << file.string() << std::endl;
        nc_close(ncid);
        return data;
    }

    int dimids[3];
    size_t T, Y, X;
    nc_inq_vardimid(ncid, varid, dimids);
    nc_inq_dimlen(ncid, dimids[0], &T);
    nc_inq_dimlen(ncid, dimids[1], &Y);
    nc_inq_dimlen(ncid, dimids[2], &X);

    int local_y_end = y_end;
    if (static_cast<size_t>(local_y_end) > Y)
        local_y_end = static_cast<int>(Y);
    size_t region_height = local_y_end - y_start;
    data.resize(T, std::vector<std::vector<double>>(region_height, std::vector<double>(X, 0.0)));

    size_t start[3] = { 0, static_cast<size_t>(y_start), 0 };
    size_t count[3] = { T, region_height, X };
    std::vector<double> buffer(T * region_height * X, 0.0);
    std::cout << "start\n";
    retval = nc_get_vara_double(ncid, varid, start, count, buffer.data());
    std::cout << "end\n";
    if (retval != NC_NOERR) {
        std::cerr << "������ ������ ����� " << file.string() << " : " << nc_strerror(retval) << std::endl;
        nc_close(ncid);
        return data;
    }
    nc_close(ncid);

    // ����������� ������ �� ������ � 3D-������
    for (size_t t = 0; t < T; t++) {
        for (size_t i = 0; i < region_height; i++) {
            for (size_t x = 0; x < X; x++) {
                size_t idx = t * region_height * X + i * X + x;
                data[t][i][x] = buffer[idx];
            }
        }
    }
    return data;
}

// ���������� ������ WaveManager::load_mariogramm_by_region � �������������� netcdf.h
std::vector<std::vector<std::vector<double>>> WaveManager::load_mariogramm_by_region(int y_start, int y_end) {

    return read_nc_file(nc_file, y_start, y_end);
}


// ������� ��� ���������� ������� �� ����� �����
int extractIndex(const fs::path& filePath) {
    // ���������� ��������� ��� ������ ������� _<�����>.nc
    std::regex regexPattern("_(\\d+)\\.nc");
    std::smatch match;
    std::string filename = filePath.filename().string();
    if (std::regex_search(filename, match, regexPattern)) {
        return std::stoi(match[1].str());
    }
    // ���� ������ �� ������, ���������� ������������ ��������,
    // ����� ���� �������� � ����� ���������������� ������.
    return std::numeric_limits<int>::max();
}

// ������� ��� ��������� ���������������� ������ ������
std::vector<fs::path> getSortedFileList(const std::string& folder) {
    std::vector<fs::path> files;

    // ���������� ��� ����� � �������� ����������
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.is_regular_file()) {
            fs::path filePath = entry.path();
            // ���������, ��� ���� ����� ���������� ".nc" � �������� ������ '_'
            if (filePath.extension() == ".nc" &&
                filePath.filename().string().find('_') != std::string::npos) {
                files.push_back(filePath);
            }
        }
    }

    // ��������� ����� �� ��������� ��������, ������������ �� ����� �����
    std::sort(files.begin(), files.end(), [](const fs::path& a, const fs::path& b) {
        return extractIndex(a) < extractIndex(b);
        });

    return files;
}


std::vector<std::vector<std::vector<std::vector<double>>>> BasisManager::get_fk_region(int y_start, int y_end) {
    std::vector<std::vector<std::vector<std::vector<double>>>> fk;
    std::vector<fs::path> files = getSortedFileList(folder);

    // ���������������� ��������� ������
    for (const auto& file : files) {
        auto file_data = read_nc_file(file, y_start, y_end);
        if (!file_data.empty()) {
            fk.push_back(file_data);
        }
    }
    return fk;
}