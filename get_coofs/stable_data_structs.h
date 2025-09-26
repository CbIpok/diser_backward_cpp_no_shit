#ifndef STABLE_DATA_STRUCTS_H
#define STABLE_DATA_STRUCTS_H

#include <string>
#include <vector>
#include <fstream>
#include <initializer_list>
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

        auto parse_size = [](const nlohmann::json& value) {
            std::vector<int> result;
            if (value.is_array()) {
                for (const auto& item : value) {
                    if (item.is_number_integer()) {
                        result.push_back(item.get<int>());
                    }
                }
            }
            else if (value.is_object()) {
                auto read = [&value](std::initializer_list<const char*> keys) {
                    for (auto key : keys) {
                        if (value.contains(key) && value[key].is_number_integer()) {
                            return value[key].get<int>();
                        }
                    }
                    return 0;
                };
                result.push_back(read({ "width", "w", "size_x", "x" }));
                result.push_back(read({ "height", "h", "size_y", "y" }));
            }
            return result;
        };

        auto parse_bounds = [](const nlohmann::json& value) {
            std::vector<int> result;
            if (value.is_array()) {
                for (const auto& item : value) {
                    if (item.is_number_integer()) {
                        result.push_back(item.get<int>());
                    }
                }
            }
            else if (value.is_object()) {
                auto read = [](const nlohmann::json& src, std::initializer_list<const char*> keys) {
                    for (auto key : keys) {
                        if (src.contains(key) && src[key].is_number_integer()) {
                            return src[key].get<int>();
                        }
                    }
                    return 0;
                };
                if (value.contains("x") && value["x"].is_object()) {
                    const auto& x_obj = value["x"];
                    result.push_back(read(x_obj, { "min", "start", "from" }));
                    result.push_back(read(x_obj, { "max", "end", "to" }));
                }
                else {
                    result.push_back(read(value, { "min_x", "x_min", "xmin", "left" }));
                    result.push_back(read(value, { "max_x", "x_max", "xmax", "right" }));
                }
                if (value.contains("y") && value["y"].is_object()) {
                    const auto& y_obj = value["y"];
                    result.push_back(read(y_obj, { "min", "start", "from" }));
                    result.push_back(read(y_obj, { "max", "end", "to" }));
                }
                else {
                    result.push_back(read(value, { "min_y", "y_min", "ymin", "bottom" }));
                    result.push_back(read(value, { "max_y", "y_max", "ymax", "top" }));
                }
            }
            return result;
        };

        if (j.contains("size")) {
            auto parsed_size = parse_size(j["size"]);
            if (!parsed_size.empty()) {
                all = parsed_size;
            }
        }

        if (all.empty() && j.contains("info") && j["info"].is_object()) {
            const auto& info = j["info"];
            if (info.contains("size")) {
                auto parsed_size = parse_size(info["size"]);
                if (!parsed_size.empty()) {
                    all = parsed_size;
                }
            }
        }

        auto assign_if_not_empty = [](std::vector<int>& target, const std::vector<int>& source) {
            if (!source.empty()) {
                target = source;
            }
        };

        if (j.contains("subduction_zone")) {
            assign_if_not_empty(subduction_bounds, parse_bounds(j["subduction_zone"]));
        }
        if (j.contains("mariogramm_zone")) {
            assign_if_not_empty(mariogramm_bounds, parse_bounds(j["mariogramm_zone"]));
        }

        if (j.contains("zones") && j["zones"].is_object()) {
            const auto& zones = j["zones"];
            if (zones.contains("subduction")) {
                assign_if_not_empty(subduction_bounds, parse_bounds(zones["subduction"]));
            }
            if (zones.contains("mariogramm")) {
                assign_if_not_empty(mariogramm_bounds, parse_bounds(zones["mariogramm"]));
            }
        }
        if (j.contains("info") && j["info"].is_object()) {
            const auto& info = j["info"];
            if (info.contains("zones") && info["zones"].is_object()) {
                const auto& info_zones = info["zones"];
                if (info_zones.contains("subduction")) {
                    assign_if_not_empty(subduction_bounds, parse_bounds(info_zones["subduction"]));
                }
                if (info_zones.contains("mariogramm")) {
                    assign_if_not_empty(mariogramm_bounds, parse_bounds(info_zones["mariogramm"]));
                }
            }
        }
    }
};

#endif // STABLE_DATA_STRUCTS_H