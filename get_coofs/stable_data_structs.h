// stable_data_structs.h
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

// точка целочисленная
using Point2i = bg::model::d2::point_xy<int>;
using Polygon2i = bg::model::polygon<Point2i>;

constexpr int scale_factor = 4;

class AreaConfigurationInfo {
public:
    int width, height;
    std::string bath_path;
    Polygon2i mariogramm_poly;

    AreaConfigurationInfo() = default;
    explicit AreaConfigurationInfo(const std::string& json_file) {
        load(json_file);
    }

    void load(const std::string& file_path) {
        // 0) Clear old points
        mariogramm_poly.outer().clear();

        std::cerr << "[LOAD] Starting to load configuration from: "
            << file_path << "\n";

        // 1) Open the file
        std::ifstream ifs(file_path);
        if (!ifs.is_open()) {
            std::cerr << "[ERROR] Failed to open file: "
                << file_path << "\n";
            return;
        }
        std::cerr << "[LOAD] File opened successfully\n";

        // 2) Read everything into a string (so we can print the contents)
        std::string content{ std::istreambuf_iterator<char>(ifs),
                             std::istreambuf_iterator<char>() };
        std::cerr << "[LOAD] Bytes read: " << content.size() << "\n";
        std::cerr << "[LOAD] Start of file (first 200 characters):\n"
            << content.substr(0, 200) << "\n---\n";

        // 3) Try to parse JSON
        nlohmann::json j;
        try {
            j = nlohmann::json::parse(content);
        }
        catch (const nlohmann::json::parse_error& e) {
            std::cerr << "[ERROR] JSON parsing error: "
                << e.what()
                << " (at byte " << e.byte << ")\n";
            return;
        }
        std::cerr << "[LOAD] JSON parsed successfully\n";

        // 4) Print which keys are at the root
        std::cerr << "[LOAD] JSON file keys:";
        for (auto it = j.begin(); it != j.end(); ++it) {
            std::cerr << " \"" << it.key() << "\"";
        }
        std::cerr << "\n";

        // 5) Check required fields
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

        // 6) Read size and bath_path
        width = j["size"]["x"].get<int>()/scale_factor;
        height = j["size"]["y"].get<int>() / scale_factor;
        bath_path = j["bath_path"].get<std::string>();
        std::cerr << "[LOAD] size = (" << width
            << "," << height << "), bath_path = \""
            << bath_path << "\"\n";

        // 7) Read and log each polygon point
        auto& ring = mariogramm_poly.outer();
        for (const auto& p : j["mariogramm_polygon"]) {
            if (!p.is_array() || p.size() != 2) {
                std::cerr << "[WARNING] Skipping invalid point\n";
                continue;
            }
            int x = p[0].get<int>() / scale_factor;
            int y = p[1].get<int>() / scale_factor;
            ring.emplace_back(x, y);
            std::cerr << "[LOAD] Added point: (" << x << "," << y << ")\n";
        }

        // 8) Explicitly close the ring if it isn't closed
        ring.push_back(ring.front());
        std::cerr << "[LOAD] Closing ring: added first point at the end\n";

        // 9) Correct (orientation, simplification, closure)
        bg::correct(mariogramm_poly);

        // 10) Final report
        std::cerr << "[LOAD] Total points in ring (including closure): "
            << ring.size() << "\n";
        for (size_t i = 0; i < ring.size(); ++i) {
            std::cerr << "  [" << i << "] ("
                << bg::get<0>(ring[i]) << ","
                << bg::get<1>(ring[i]) << ")\n";
        }
        std::cerr << "[LOAD] Configuration loading complete\n";
    }

    void draw(cv::Mat& img,
        const cv::Scalar& color,
        int thickness) const
    {
        // 1) собираем все точки в vector<Point>
        std::vector<cv::Point> pts;
        pts.reserve(mariogramm_poly.outer().size());
        for (auto const& p : mariogramm_poly.outer()) {
            int x = static_cast<int>(bg::get<0>(p));
            int y = static_cast<int>(bg::get<1>(p));
            pts.emplace_back(x, y);
        }
        if (pts.size() < 2) return;

        // 2) OpenCV ожидает InputArrayOfArrays — оборачиваем
        std::vector<std::vector<cv::Point>> contours;
        contours.emplace_back(std::move(pts));

        // 3) рисуем замкнутый контур
        cv::polylines(img, contours, /*isClosed=*/true, color, thickness);
    }

};

#endif // STABLE_DATA_STRUCTS_H
