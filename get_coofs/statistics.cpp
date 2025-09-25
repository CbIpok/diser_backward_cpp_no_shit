#include "statistics.h"

#include "approx_orto.h"

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
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

    std::size_t max_points_per_chunk = MAX_MEMORY_BYTES / bytes_per_point;
    if (max_points_per_chunk == 0) {
        max_points_per_chunk = 1;
    }

    std::map<int, std::vector<CoefficientData>> aggregated_rows;
    uint64_t total_load_ms = 0;
    uint64_t total_proc_ms = 0;
    bool had_error = false;

    for (std::size_t chunk_begin = 0; chunk_begin < inside_points.size() && !had_error; chunk_begin += max_points_per_chunk) {
        std::size_t chunk_end = std::min(chunk_begin + max_points_per_chunk, inside_points.size());
        std::size_t chunk_size = chunk_end - chunk_begin;
        const Point2i* chunk_points = inside_points.data() + chunk_begin;

        std::vector<double> wave_buffer;
        std::vector<std::vector<double>> basis_buffer;

        std::cout << "LOAD_START thread=-1 points=" << chunk_size << "\n";
        auto load_start = std::chrono::steady_clock::now();
        if (!wave_manager.load_points(chunk_points, chunk_size, wave_buffer)) {
            had_error = true;
            break;
        }
        if (!basis_manager.load_points(chunk_points, chunk_size, basis_buffer)) {
            had_error = true;
            break;
        }
        auto load_end = std::chrono::steady_clock::now();
        auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
        total_load_ms += static_cast<uint64_t>(load_ms);
        std::cout << "LOAD_END thread=-1 points=" << chunk_size << " ms=" << load_ms << "\n";

        unsigned active_threads = std::min<unsigned>(thread_count, static_cast<unsigned>(chunk_size));
        if (active_threads == 0) {
            active_threads = 1;
        }

        std::vector<std::map<int, std::vector<CoefficientData>>> local_rows(active_threads);
        std::vector<uint64_t> local_proc_ms(active_threads, 0);

        std::size_t base_chunk = chunk_size / active_threads;
        std::size_t remainder = chunk_size % active_threads;
        std::size_t offset = 0;

        std::vector<std::thread> workers;
        workers.reserve(active_threads);

        for (unsigned t = 0; t < active_threads; ++t) {
            std::size_t take = base_chunk + (t < remainder ? 1 : 0);
            std::size_t local_begin = offset;
            std::size_t local_end = offset + take;
            offset = local_end;

            workers.emplace_back([&, t, local_begin, local_end, chunk_points]() {
                if (local_begin >= local_end) {
                    return;
                }

                std::size_t local_count = local_end - local_begin;
                std::cout << "PROC_START thread=" << t << " points=" << local_count << "\n";
                auto proc_start = std::chrono::steady_clock::now();

                for (std::size_t idx = local_begin; idx < local_end; ++idx) {
                    const Point2i& pt = chunk_points[idx];
                    Eigen::Map<const Eigen::VectorXd> wave_vec(&wave_buffer[idx * wave_info.t], static_cast<Eigen::Index>(wave_info.t));
                    Eigen::MatrixXd B(n_basis, static_cast<Eigen::Index>(wave_info.t));
                    for (int b = 0; b < n_basis; ++b) {
                        Eigen::Map<const Eigen::VectorXd> basis_vec(&basis_buffer[static_cast<std::size_t>(b)][idx * wave_info.t], static_cast<Eigen::Index>(wave_info.t));
                        B.row(static_cast<Eigen::Index>(b)) = basis_vec.transpose();
                    }

                    Eigen::VectorXd coefs = approximate_with_non_orthogonal_basis_orto(wave_vec, B);
                    Eigen::VectorXd approx = B.transpose() * coefs;
                    double err = std::sqrt((wave_vec - approx).squaredNorm() / static_cast<double>(wave_info.t));

                    local_rows[t][bg::get<1>(pt)].push_back({ pt, std::move(coefs), err });
                }

                auto proc_end = std::chrono::steady_clock::now();
                auto proc_ms = std::chrono::duration_cast<std::chrono::milliseconds>(proc_end - proc_start).count();
                local_proc_ms[t] += static_cast<uint64_t>(proc_ms);
                std::cout << "PROC_END thread=" << t << " points=" << local_count << " ms=" << proc_ms << "\n";
            });
        }

        for (auto& worker_thread : workers) {
            worker_thread.join();
        }

        for (unsigned t = 0; t < active_threads; ++t) {
            total_proc_ms += local_proc_ms[t];
            for (auto& [y, row] : local_rows[t]) {
                auto& dst = aggregated_rows[y];
                dst.insert(dst.end(), std::make_move_iterator(row.begin()), std::make_move_iterator(row.end()));
            }
        }
    }

    uint64_t total_points = static_cast<uint64_t>(inside_points.size());
    std::cout << "SUMMARY points_total=" << total_points
        << " threads=" << thread_count
        << " load_ms_total=" << total_load_ms
        << " proc_ms_total=" << total_proc_ms << "\n";

    if (had_error) {
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
