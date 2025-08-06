#ifndef STABLE_DATA_STRUCTS_H
#define STABLE_DATA_STRUCTS_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "json.hpp"

class AreaConfigurationInfo {
public:
    std::vector<int> all;              // ������� �������, ��������, [width, height]
    std::vector<int> subduction_bounds;
    std::vector<int> mariogramm_bounds;

    AreaConfigurationInfo() = default;
    explicit AreaConfigurationInfo(const std::string& json_file) {
        load(json_file);
    }

    void load(const std::string& file_path) {
        std::ifstream ifs(file_path);
        if (!ifs.is_open()) {
            std::cerr << "������ �������� �����: " << file_path << std::endl;
            return;
        }
        nlohmann::json j;
        ifs >> j;
        all = j["size"].get<std::vector<int>>();
        subduction_bounds = j["subduction_zone"].get<std::vector<int>>();
        mariogramm_bounds = j["mariogramm_zone"].get<std::vector<int>>();
    }
};

#endif // STABLE_DATA_STRUCTS_H