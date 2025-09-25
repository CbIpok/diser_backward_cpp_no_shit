#ifndef MANAGERS_H
#define MANAGERS_H

#include <string>
#include <vector>
#include <filesystem>
#include "stable_data_structs.h"

struct NetCDFVariableInfo {
    std::size_t t = 0;
    std::size_t y = 0;
    std::size_t x = 0;
};

class BasisManager {
public:
    explicit BasisManager(const std::string& folder_);

    int basis_count() const;
    const NetCDFVariableInfo& describe() const;
    bool load_points(const Point2i* points, std::size_t count, std::vector<std::vector<double>>& out) const;

private:
    std::string folder;
    std::vector<std::filesystem::path> basis_files;
    NetCDFVariableInfo info{};
    bool has_info = false;
};

class WaveManager {
public:
    explicit WaveManager(const std::string& nc_file_);

    const NetCDFVariableInfo& describe() const;
    bool valid() const;
    bool load_points(const Point2i* points, std::size_t count, std::vector<double>& out) const;

private:
    std::string nc_file;
    NetCDFVariableInfo info{};
    bool has_info = false;
};

#endif // MANAGERS_H