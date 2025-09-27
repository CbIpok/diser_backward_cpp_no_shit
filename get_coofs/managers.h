#ifndef MANAGERS_H
#define MANAGERS_H

#include <string>
#include <vector>
#include "stable_data_structs.h"

class BasisManager {
public:
    std::string folder;

    explicit BasisManager(const std::string& folder_) : folder(folder_) {}

    std::vector<std::vector<std::vector<std::vector<double>>>> get_fk_region(int y_start, int y_end);

    bool describe(size_t& dataset_count, size_t& time_dim, size_t& y_dim, size_t& x_dim) const;
};

class WaveManager {
public:
    std::string nc_file;

    explicit WaveManager(const std::string& nc_file_) : nc_file(nc_file_) {}

    std::vector<std::vector<std::vector<double>>> load_mariogramm_by_region(int y_start, int y_end);

    bool describe(size_t& time_dim, size_t& y_dim, size_t& x_dim) const;
};

#endif // MANAGERS_H

