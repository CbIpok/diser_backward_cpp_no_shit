#ifndef MANAGERS_H
#define MANAGERS_H

#include <string>
#include <cstddef>
#include <vector>
#include "stable_data_structs.h"


class BasisManager {
public:
    std::string folder;

    explicit BasisManager(const std::string& folder_) : folder(folder_) {}

    std::vector<std::vector<std::vector<std::vector<double>>>> get_fk_region(int y_start, int y_end);
    std::size_t basis_count() const;
    bool get_dimensions(std::size_t& T, std::size_t& Y, std::size_t& X) const;
};

class WaveManager {
public:
    std::string nc_file;

    explicit WaveManager(const std::string& nc_file_) : nc_file(nc_file_) {}

    std::vector<std::vector<std::vector<double>>> load_mariogramm_by_region(int y_start, int y_end);
    bool get_dimensions(std::size_t& T, std::size_t& Y, std::size_t& X) const;
};

#endif // MANAGERS_H