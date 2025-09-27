#ifndef STABLE_DATA_STRUCTS_H
#define STABLE_DATA_STRUCTS_H

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include "json.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace bg = boost::geometry;

using Point2i = bg::model::d2::point_xy<int>;
using Polygon2i = bg::model::polygon<Point2i>;

constexpr int scale_factor = 4;

class AreaConfigurationInfo {
public:
    int width = 0;
    int height = 0;
    std::string bath_path;
    Polygon2i mariogramm_poly;

    AreaConfigurationInfo() = default;
    explicit AreaConfigurationInfo(const std::string& json_file) {
        load(json_file);
    }

    void load(const std::string& file_path) {
        mariogramm_poly.outer().clear();

        std::ifstream ifs(file_path);
        if (!ifs.is_open()) {
            std::cerr << "[ERROR] Failed to open file: " << file_path << '\n';
            return;
        }

        nlohmann::json j;
        try {
            ifs >> j;
        }
        catch (const nlohmann::json::parse_error& e) {
            std::cerr << "[ERROR] JSON parsing error: " << e.what() << '\n';
            return;
        }

        if (!j.contains("size") || !j["size"].is_object()) {
            std::cerr << "[ERROR] Missing \"size\" field or it is not an object\n";
            return;
        }
        if (!j["size"].contains("x") || !j["size"].contains("y")) {
            std::cerr << "[ERROR] \"size\" object missing \"x\" or \"y\"\n";
            return;
        }
        if (!j.contains("bath_path") || !j["bath_path"].is_string()) {
            std::cerr << "[ERROR] Missing \"bath_path\" field or it is not a string\n";
            return;
        }
        if (!j.contains("mariogramm_polygon") || !j["mariogramm_polygon"].is_array()) {
            std::cerr << "[ERROR] Missing \"mariogramm_polygon\" field or it is not an array\n";
            return;
        }

        width = j["size"]["x"].get<int>() / scale_factor;
        height = j["size"]["y"].get<int>() / scale_factor;
        bath_path = j["bath_path"].get<std::string>();

        auto& ring = mariogramm_poly.outer();
        for (const auto& p : j["mariogramm_polygon"]) {
            if (!p.is_array() || p.size() != 2) {
                std::cerr << "[WARNING] Skipping invalid point entry\n";
                continue;
            }
            int x = p[0].get<int>() / scale_factor;
            int y = p[1].get<int>() / scale_factor;
            ring.emplace_back(x, y);
        }

        if (!ring.empty()) {
            ring.push_back(ring.front());
            bg::correct(mariogramm_poly);
        }
    }

    void draw(cv::Mat& img, const cv::Scalar& color, int thickness) const {
        std::vector<cv::Point> pts;
        pts.reserve(mariogramm_poly.outer().size());
        for (auto const& p : mariogramm_poly.outer()) {
            int x = static_cast<int>(bg::get<0>(p));
            int y = static_cast<int>(bg::get<1>(p));
            pts.emplace_back(x, y);
        }
        if (pts.size() < 2) {
            return;
        }

        std::vector<std::vector<cv::Point>> contours;
        contours.emplace_back(std::move(pts));

        cv::polylines(img, contours, true, color, thickness);
    }
};

#endif // STABLE_DATA_STRUCTS_H
