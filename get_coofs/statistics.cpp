#include "statistics.h"

#include "approx_orto.h"

#include <Eigen/Dense>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <mutex>
#include <thread>

namespace bg = boost::geometry;

namespace {

struct PointLess {
    bool operator()(const Point2i& lhs, const Point2i& rhs) const {
        int y_lhs = bg::get<1>(lhs);
        int y_rhs = bg::get<1>(rhs);
        if (y_lhs != y_rhs) {
            return y_lhs < y_rhs;
        }
        return bg::get<0>(lhs) < bg::get<0>(rhs);
    }
};

constexpr std::size_t MAX_MEMORY_BYTES = 45ULL * 1024ULL * 1024ULL * 1024ULL;

} // namespace

void calculate_statistics(const std::string& root_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config,
    CoeffMatrix& statistics_orto) {

    statistics_orto.clear();

    int minY = area_config.height;
    int maxY = 0;
    int minX = area_config.width;
    int maxX = 0;

    for (auto const& p : area_config.mariogramm_poly.outer()) {
        minX = std::min(minX, static_cast<int>(bg::get<0>(p)));
        maxX = std::max(maxX, static_cast<int>(bg::get<0>(p)));
        minY = std::min(minY, static_cast<int>(bg::get<1>(p)));
        maxY = std::max(maxY, static_cast<int>(bg::get<1>(p)));
    }

    minX = std::max(minX, 0);
    minY = std::max(minY, 0);
    maxX = std::min(maxX, area_config.width - 1);
    maxY = std::min(maxY, area_config.height - 1);

    if (minX > maxX || minY > maxY) {
        return;
    }

    std::vector<Point2i> inside_points;
    const std::size_t bbox_width = static_cast<std::size_t>(maxX - minX + 1);
    const std::size_t bbox_height = static_cast<std::size_t>(maxY - minY + 1);
    inside_points.reserve(bbox_width * bbox_height);

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            Point2i pt{ x, y };
            if (bg::within(pt, area_config.mariogramm_poly)) {
                inside_points.push_back(pt);
            }
        }
    }

    if (inside_points.empty()) {
        return;
    }

    std::sort(inside_points.begin(), inside_points.end(), PointLess{});

    BasisManager basis_manager(root_folder + "/" + bath + "/" + basis);
    WaveManager wave_manager(root_folder + "/" + bath + "/" + wave + ".nc");

    if (!wave_manager.valid()) {
        std::cerr << "Wave manager is not ready; aborting statistics calculation" << std::endl;
        return;
    }

    const NetCDFVariableInfo& wave_info = wave_manager.describe();
    const NetCDFVariableInfo& basis_info = basis_manager.describe();
    const int n_basis = basis_manager.basis_count();

    if (wave_info.t == 0 || n_basis <= 0 || basis_info.t == 0) {
        std::cerr << "Insufficient NetCDF metadata to continue" << std::endl;
        return;
    }

    if (basis_info.t != 0 && basis_info.t != wave_info.t) {
        std::cerr << "Warning: basis time dimension does not match wave data" << std::endl;
    }

    std::size_t bytes_per_point = static_cast<std::size_t>(n_basis + 1) * wave_info.t * sizeof(double);
    if (bytes_per_point == 0) {
        return;
    }

    unsigned thread_count = std::max(1u, std::thread::hardware_concurrency());
    thread_count = std::min<unsigned>(thread_count, static_cast<unsigned>(inside_points.size()));
    if (thread_count == 0) {
        thread_count = 1;
    }

    std::size_t per_thread_limit = thread_count > 0 ? MAX_MEMORY_BYTES / thread_count : MAX_MEMORY_BYTES;
    if (per_thread_limit == 0) {
        per_thread_limit = MAX_MEMORY_BYTES;
    }
    std::size_t max_points_per_subchunk = per_thread_limit / bytes_per_point;
    if (max_points_per_subchunk == 0) {
        max_points_per_subchunk = 1;
    }

    std::vector<std::pair<std::size_t, std::size_t>> thread_ranges(thread_count);
    std::size_t base_chunk = inside_points.size() / thread_count;
    std::size_t remainder = inside_points.size() % thread_count;
    std::size_t offset = 0;
    for (unsigned t = 0; t < thread_count; ++t) {
        std::size_t chunk = base_chunk + (t < remainder ? 1 : 0);
        thread_ranges[t] = { offset, offset + chunk };
        offset += chunk;
    }

    std::map<int, std::vector<CoefficientData>> aggregated_rows;
    std::mutex aggregated_mutex;
    std::atomic<uint64_t> total_load_ms{ 0 };
    std::atomic<uint64_t> total_proc_ms{ 0 };
    std::atomic<bool> had_error{ false };

    auto worker = [&](unsigned thread_index, std::size_t begin, std::size_t end) {
        if (begin >= end) {
            return;
        }

        std::map<int, std::vector<CoefficientData>> local_rows;
        uint64_t local_load = 0;
        uint64_t local_proc = 0;

        std::size_t current = begin;
        while (current < end && !had_error.load(std::memory_order_relaxed)) {
            std::size_t remaining = end - current;
            std::size_t take = std::min(remaining, max_points_per_subchunk);
            const Point2i* sub_points = inside_points.data() + current;

            std::vector<double> wave_buffer;
            std::vector<std::vector<double>> basis_buffer;

            std::cout << "LOAD_START thread=" << thread_index << " points=" << take << "\n";
            auto load_start = std::chrono::steady_clock::now();
            if (!wave_manager.load_points(sub_points, take, wave_buffer)) {
                had_error.store(true, std::memory_order_relaxed);
                break;
            }
            if (!basis_manager.load_points(sub_points, take, basis_buffer)) {
                had_error.store(true, std::memory_order_relaxed);
                break;
            }
            auto load_end = std::chrono::steady_clock::now();
            auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
            local_load += static_cast<uint64_t>(load_ms);
            std::cout << "LOAD_END thread=" << thread_index << " points=" << take << " ms=" << load_ms << "\n";

            std::cout << "PROC_START thread=" << thread_index << " points=" << take << "\n";
            auto proc_start = std::chrono::steady_clock::now();

            for (std::size_t idx = 0; idx < take; ++idx) {
                const Point2i& pt = sub_points[idx];
                Eigen::Map<const Eigen::VectorXd> wave_vec(&wave_buffer[idx * wave_info.t], static_cast<Eigen::Index>(wave_info.t));
                Eigen::MatrixXd B(n_basis, static_cast<Eigen::Index>(wave_info.t));
                for (int b = 0; b < n_basis; ++b) {
                    Eigen::Map<const Eigen::VectorXd> basis_vec(&basis_buffer[static_cast<std::size_t>(b)][idx * wave_info.t], static_cast<Eigen::Index>(wave_info.t));
                    B.row(static_cast<Eigen::Index>(b)) = basis_vec.transpose();
                }

                Eigen::VectorXd coefs = approximate_with_non_orthogonal_basis_orto(wave_vec, B);
                Eigen::VectorXd approx = B.transpose() * coefs;
                double err = std::sqrt((wave_vec - approx).squaredNorm() / static_cast<double>(wave_info.t));

                local_rows[bg::get<1>(pt)].push_back({ pt, std::move(coefs), err });
            }

            auto proc_end = std::chrono::steady_clock::now();
            auto proc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(proc_end - proc_start).count();
            local_proc += static_cast<uint64_t>(proc_ms);
            std::cout << "PROC_END thread=" << thread_index << " points=" << take << " ms=" << proc_ms << "\n";

            current += take;
        }

        {
            std::lock_guard<std::mutex> lock(aggregated_mutex);
            for (auto& [y, row] : local_rows) {
                auto& dst = aggregated_rows[y];
                dst.insert(dst.end(), std::make_move_iterator(row.begin()), std::make_move_iterator(row.end()));
            }
        }

        total_load_ms.fetch_add(local_load, std::memory_order_relaxed);
        total_proc_ms.fetch_add(local_proc, std::memory_order_relaxed);
    };

    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    for (unsigned t = 0; t < thread_count; ++t) {
        auto [begin, end_range] = thread_ranges[t];
        workers.emplace_back(worker, t, begin, end_range);
    }

    for (auto& thread : workers) {
        thread.join();
    }

    uint64_t total_points = static_cast<uint64_t>(inside_points.size());
    uint64_t load_total = total_load_ms.load(std::memory_order_relaxed);
    uint64_t proc_total = total_proc_ms.load(std::memory_order_relaxed);
    std::cout << "SUMMARY points_total=" << total_points
        << " threads=" << thread_count
        << " load_ms_total=" << load_total
        << " proc_ms_total=" << proc_total << "\n";

    if (had_error.load(std::memory_order_relaxed)) {
        return;
    }

    for (auto& [y, row] : aggregated_rows) {
        std::sort(row.begin(), row.end(), [](const CoefficientData& lhs, const CoefficientData& rhs) {
            return bg::get<0>(lhs.pt) < bg::get<0>(rhs.pt);
        });
        statistics_orto.push_back(std::move(row));
    }
}

void save_coefficients_json(const std::string& filename, const CoeffMatrix& coeffs) {
    nlohmann::json j;
    for (size_t row = 0; row < coeffs.size(); ++row) {
        for (size_t col = 0; col < coeffs[row].size(); ++col) {
            std::string key = "[" + std::to_string(bg::get<0>(coeffs[row][col].pt)) + "," + std::to_string(bg::get<1>(coeffs[row][col].pt)) + "]";

            std::vector<double> vec(coeffs[row][col].coefs.data(),
                coeffs[row][col].coefs.data() + coeffs[row][col].coefs.size());
            double error = coeffs[row][col].aprox_error;
            j[key] = { {"coefs", vec}, {"aprox_error", error} };
        }
    }
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Failed to open output file: " << filename << "\n";
        return;
    }
    ofs << j.dump(4);
    ofs.close();
    std::cout << "saved: " << filename << "\n";
}

void save_and_plot_statistics(const std::string& root_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config) {
    CoeffMatrix statistics_orto;
    calculate_statistics(root_folder, bath, wave, basis, area_config, statistics_orto);

    std::string filename_orto = "case_statistics_" + basis + bath + wave + "_o.json";

    save_coefficients_json(filename_orto, statistics_orto);
}
