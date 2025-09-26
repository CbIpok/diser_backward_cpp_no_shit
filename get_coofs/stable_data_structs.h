#ifndef STABLE_DATA_STRUCTS_H
#define STABLE_DATA_STRUCTS_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "json.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>

namespace bg = boost::geometry;
using AreaPoint = bg::model::d2::point_xy<double>;
using AreaPolygon = bg::model::polygon<AreaPoint>;

class AreaConfigurationInfo {
public:
    std::vector<int> all;              // размеры области, например, [width, height]
    std::vector<int> subduction_bounds;
    std::vector<int> mariogramm_bounds;
    AreaPolygon mariogramm_poly;

    AreaConfigurationInfo() = default;
    explicit AreaConfigurationInfo(const std::string& json_file) {
        load(json_file);
    }

    void load(const std::string& file_path) {
        std::ifstream ifs(file_path);
        if (!ifs.is_open()) {
            std::cerr << "Ошибка открытия файла: " << file_path << std::endl;
            return;
        }
        nlohmann::json j;
        ifs >> j;
        if (j.contains("size")) {
            all = j["size"].get<std::vector<int>>();
        }
        if (j.contains("subduction_zone")) {
            subduction_bounds = j["subduction_zone"].get<std::vector<int>>();
        }
        if (j.contains("mariogramm_zone")) {
            mariogramm_bounds = j["mariogramm_zone"].get<std::vector<int>>();
        }
        if (j.contains("mariogramm_poly")) {
            mariogramm_poly.outer().clear();
            for (const auto& point : j["mariogramm_poly"]) {
                if (point.size() < 2) {
                    continue;
                }
                double x = point[0].get<double>();
                double y = point[1].get<double>();
                mariogramm_poly.outer().emplace_back(x, y);
            }
            if (!mariogramm_poly.outer().empty()) {
                const auto& first = mariogramm_poly.outer().front();
                const auto& last = mariogramm_poly.outer().back();
                if (!bg::equals(first, last)) {
                    mariogramm_poly.outer().push_back(first);
                }
                bg::correct(mariogramm_poly);
            }
        }
    }
};

#endif // STABLE_DATA_STRUCTS_H
