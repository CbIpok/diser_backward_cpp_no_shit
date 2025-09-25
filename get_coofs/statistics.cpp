// statistics.cpp
#include "statistics.h"
#include "approx_orto.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <limits>
#include <utility>
#include <thread>
#include <iterator>
#include <chrono>
#include <memory>
#include "json.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>

using json = nlohmann::json;
namespace bg = boost::geometry;
using Point2i = bg::model::d2::point_xy<int>;


int count_from_name(const std::string& name) {
  
    std::size_t underscorePos = name.find('_');
    if (underscorePos != std::string::npos) {
     
        std::string numberPart = name.substr(underscorePos + 1);
     
        return std::stoi(numberPart);
    }
   
    return 0;
}


void calculate_statistics(const std::string& root_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config,
    CoeffMatrix& statistics_orto) {

    // 1) вычисляем границы полигона
    int minY = area_config.height, maxY = 0;
    int minX = area_config.width, maxX = 0;
    for (auto const& p : area_config.mariogramm_poly.outer()) {
        minX = std::min(minX, int(bg::get<0>(p)));
        maxX = std::max(maxX, int(bg::get<0>(p)));
        minY = std::min(minY, int(bg::get<1>(p)));
        maxY = std::max(maxY, int(bg::get<1>(p)));
    }
    BasisManager basis_manager(root_folder + "/" + bath + "/" + basis);
    WaveManager wave_manager(root_folder + "/" + bath + "/" + wave + ".nc");

    if (!wave_manager.valid()) {
        std::cerr << "failed to open wave dataset " << wave_manager.path() << '\n';
        return;
    }
    if (!basis_manager.valid()) {
        std::cerr << "failed to open basis datasets in " << basis_manager.directory() << '\n';
        return;
    }

    std::size_t wave_T = 0, wave_Y = 0, wave_X = 0;
    if (!wave_manager.get_dimensions(wave_T, wave_Y, wave_X)) {
        std::cerr << "failed to query wave dimensions for " << wave_manager.path() << '\n';
        return;
    }

    std::size_t basis_T = 0, basis_Y = 0, basis_X = 0;
    if (!basis_manager.get_dimensions(basis_T, basis_Y, basis_X)) {
        std::cerr << "failed to query basis dimensions for " << basis_manager.directory() << '\n';
        return;
    }

    if (basis_T != wave_T || basis_Y != wave_Y || basis_X != wave_X) {
        std::cerr << "basis dataset dimensions do not match wave dataset" << '\n';
        return;
    }

    std::size_t basis_count = basis_manager.basis_count();
    if (basis_count == 0) {
        std::cerr << "no basis files found in " << basis_manager.directory() << '\n';
        return;
    }

    int T = static_cast<int>(wave_T);
    int W = static_cast<int>(wave_X);
    int data_height = static_cast<int>(wave_Y);
    int n_basis = static_cast<int>(basis_count);

    if (T <= 0 || W <= 0 || data_height <= 0) {
        std::cerr << "invalid data dimensions" << '\n';
        return;
    }

    minX = std::max(0, minX);
    maxX = std::min(maxX, W - 1);
    minY = std::max(0, minY);
    maxY = std::min(maxY, data_height - 1);
    if (minX > maxX || minY > maxY) {
        std::cerr << "polygon bounds do not intersect data domain" << '\n';
        return;
    }

    struct RowPoints {
        int y = 0;
        std::vector<int> xs;
    };

    std::vector<Point2i> points;
    std::vector<RowPoints> rows;
    points.reserve(static_cast<std::size_t>(maxY - minY + 1)
        * static_cast<std::size_t>(maxX - minX + 1));
    rows.reserve(static_cast<std::size_t>(maxY - minY + 1));

    for (int y = minY; y <= maxY; ++y) {
        RowPoints row;
        row.y = y;
        for (int x = minX; x <= maxX; ++x) {
            Point2i pt(x, y);
            if (bg::within(pt, area_config.mariogramm_poly)) {
                row.xs.push_back(x);
                points.emplace_back(pt);
            }
        }
        if (!row.xs.empty()) {
            rows.push_back(std::move(row));
        }
    }

    if (points.empty()) {
        return;
    }

    std::vector<std::size_t> row_point_prefix(rows.size() + 1, 0);
    for (std::size_t i = 0; i < rows.size(); ++i) {
        row_point_prefix[i + 1] = row_point_prefix[i] + rows[i].xs.size();
    }

    constexpr std::size_t MAX_MEMORY_BYTES = 45ULL * 1024ULL * 1024ULL * 1024ULL; // 45 GB
    std::size_t bytes_per_cell = static_cast<std::size_t>(n_basis + 1)
        * static_cast<std::size_t>(T) * sizeof(double);

    std::size_t max_cells = bytes_per_cell > 0 ? MAX_MEMORY_BYTES / bytes_per_cell : 0;
    if (max_cells == 0) {
        max_cells = 1;
    }
    if (max_cells > 1) {
        // Double buffering keeps one chunk in compute while the next loads, so
        // halve the memory target to honour the overall limit.
        max_cells = std::max<std::size_t>(1, max_cells / 2);
    }

    struct ChunkPlan {
        std::size_t row_start = 0;
        std::size_t row_end = 0;
        int y_start = 0;
        std::size_t y_count = 0;
        int x_start = 0;
        std::size_t x_count = 0;
        std::size_t point_count = 0;
    };

    struct LoadedChunk {
        ChunkPlan plan;
        std::vector<Point2i> points;
        std::vector<std::vector<std::vector<double>>> wave;
        std::vector<std::vector<std::vector<std::vector<double>>>> fk;
        double load_seconds = 0.0;
    };

    auto plan_chunk = [&](std::size_t row_start) -> ChunkPlan {
        ChunkPlan plan;
        if (row_start >= rows.size()) {
            return plan;
        }

        const auto& first_row = rows[row_start];
        if (first_row.xs.empty()) {
            plan.row_start = row_start;
            plan.row_end = row_start + 1;
            plan.y_start = first_row.y;
            plan.y_count = 1;
            plan.x_start = 0;
            plan.x_count = 0;
            plan.point_count = 0;
            return plan;
        }

        int x_min = first_row.xs.front();
        int x_max = first_row.xs.back();
        int y_start_row = first_row.y;
        int y_end_row = first_row.y;
        std::size_t row_end = row_start + 1;
        std::size_t point_count = first_row.xs.size();

        while (row_end < rows.size()) {
            const auto& prev_row = rows[row_end - 1];
            const auto& next_row = rows[row_end];
            if (next_row.y != prev_row.y + 1) {
                break;
            }
            if (next_row.xs.empty()) {
                ++row_end;
                continue;
            }
            int candidate_x_min = std::min(x_min, next_row.xs.front());
            int candidate_x_max = std::max(x_max, next_row.xs.back());
            int candidate_y_end = next_row.y;
            std::size_t candidate_y_count = static_cast<std::size_t>(candidate_y_end - y_start_row + 1);
            std::size_t candidate_x_count = static_cast<std::size_t>(candidate_x_max - candidate_x_min + 1);
            std::size_t candidate_cells = candidate_y_count * candidate_x_count;
            if (candidate_cells > max_cells) {
                break;
            }
            x_min = candidate_x_min;
            x_max = candidate_x_max;
            y_end_row = candidate_y_end;
            point_count += next_row.xs.size();
            ++row_end;
        }

        plan.row_start = row_start;
        plan.row_end = row_end;
        plan.y_start = y_start_row;
        plan.y_count = static_cast<std::size_t>(y_end_row - y_start_row + 1);
        plan.x_start = x_min;
        plan.x_count = static_cast<std::size_t>(x_max - x_min + 1);
        plan.point_count = point_count;
        return plan;
    };

    auto load_chunk = [&](const ChunkPlan& plan) {
        LoadedChunk chunk;
        chunk.plan = plan;
        if (plan.row_start >= plan.row_end || plan.point_count == 0) {
            return chunk;
        }

        chunk.points.reserve(plan.point_count);
        for (std::size_t row_idx = plan.row_start; row_idx < plan.row_end; ++row_idx) {
            const auto& row = rows[row_idx];
            for (int x : row.xs) {
                chunk.points.emplace_back(x, row.y);
            }
        }

        if (plan.y_count > static_cast<std::size_t>(std::numeric_limits<int>::max())
            || plan.x_count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            std::cerr << "chunk dimensions exceed int range" << '\n';
            chunk.points.clear();
            return chunk;
        }

        int y_count_int = static_cast<int>(plan.y_count);
        int x_count_int = static_cast<int>(plan.x_count);

        auto load_start = std::chrono::steady_clock::now();
        chunk.wave = wave_manager.load_block(plan.y_start, y_count_int, plan.x_start, x_count_int);
        chunk.fk = basis_manager.load_block(plan.y_start, y_count_int, plan.x_start, x_count_int);
        auto load_end = std::chrono::steady_clock::now();
        chunk.load_seconds = std::chrono::duration<double>(load_end - load_start).count();
        return chunk;
    };

    auto process_chunk = [&](const LoadedChunk& chunk,
        std::size_t chunk_start,
        std::size_t chunk_end,
        const std::chrono::steady_clock::time_point& chunk_cycle_start) {
        if (chunk.points.empty()) {
            return;
        }

        if (chunk.wave.size() != static_cast<std::size_t>(T)) {
            return;
        }
        for (const auto& plane : chunk.wave) {
            if (plane.size() != chunk.plan.y_count) {
                return;
            }
            for (const auto& row : plane) {
                if (row.size() != chunk.plan.x_count) {
                    return;
                }
            }
        }

        if (chunk.fk.size() != static_cast<std::size_t>(n_basis)) {
            return;
        }
        for (const auto& basis_planes : chunk.fk) {
            if (basis_planes.size() != static_cast<std::size_t>(T)) {
                return;
            }
            for (const auto& plane : basis_planes) {
                if (plane.size() != chunk.plan.y_count) {
                    return;
                }
                for (const auto& row : plane) {
                    if (row.size() != chunk.plan.x_count) {
                        return;
                    }
                }
            }
        }

        std::size_t chunk_size = chunk.points.size();
        if (chunk_size == 0) {
            return;
        }

        constexpr int THREADS = 48;
        int thread_count = std::min<int>(THREADS, static_cast<int>(chunk_size));
        if (thread_count <= 0) {
            return;
        }

        std::size_t points_per_thread = (chunk_size + static_cast<std::size_t>(thread_count) - 1)
            / static_cast<std::size_t>(thread_count);

        std::vector<std::vector<CoefficientData>> thread_local_results(static_cast<std::size_t>(thread_count));
        std::vector<std::thread> threads;
        threads.reserve(thread_count);

        for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
            std::size_t start_idx = static_cast<std::size_t>(thread_idx) * points_per_thread;
            if (start_idx >= chunk_size) {
                break;
            }
            std::size_t end_idx = std::min(chunk_size, start_idx + points_per_thread);

            threads.emplace_back([&, thread_idx, start_idx, end_idx]() {
                auto& local = thread_local_results[static_cast<std::size_t>(thread_idx)];
                local.reserve(end_idx - start_idx);

                Eigen::VectorXd wave_vec(T);
                Eigen::MatrixXd B(n_basis, T);

                for (std::size_t idx = start_idx; idx < end_idx; ++idx) {
                    const auto& pt = chunk.points[idx];
                    int local_y = bg::get<1>(pt) - chunk.plan.y_start;
                    int local_x = bg::get<0>(pt) - chunk.plan.x_start;
                    if (local_y < 0 || local_x < 0) {
                        continue;
                    }
                    if (local_y >= static_cast<int>(chunk.plan.y_count)
                        || local_x >= static_cast<int>(chunk.plan.x_count)) {
                        continue;
                    }

                    for (int t = 0; t < T; ++t) {
                        wave_vec[t] = chunk.wave[static_cast<std::size_t>(t)]
                                              [static_cast<std::size_t>(local_y)]
                                              [static_cast<std::size_t>(local_x)];
                    }

                    for (int b = 0; b < n_basis; ++b) {
                        for (int t = 0; t < T; ++t) {
                            B(b, t) = chunk.fk[static_cast<std::size_t>(b)]
                                               [static_cast<std::size_t>(t)]
                                               [static_cast<std::size_t>(local_y)]
                                               [static_cast<std::size_t>(local_x)];
                        }
                    }

                    auto coefs = approximate_with_non_orthogonal_basis_orto(wave_vec, B);
                    Eigen::VectorXd approx = B.transpose() * coefs;
                    double err = std::sqrt((wave_vec - approx).squaredNorm() / T);

                    local.push_back({ chunk.points[idx], coefs, err });
                }
            });
        }

        for (auto& thread : threads) {
            if (thread.joinable()) {
                thread.join();
            }
        }

        std::vector<CoefficientData> chunk_result;
        chunk_result.reserve(chunk_size);
        for (auto& local : thread_local_results) {
            std::move(local.begin(), local.end(), std::back_inserter(chunk_result));
        }

        if (!chunk_result.empty()) {
            statistics_orto.push_back(std::move(chunk_result));
        }

        auto chunk_cycle_end = std::chrono::steady_clock::now();
        auto chunk_cycle_seconds = std::chrono::duration<double>(chunk_cycle_end - chunk_cycle_start);
        std::cout << "Chunk processing cycle for points " << chunk_start << "-" << chunk_end
                  << " took " << chunk_cycle_seconds.count() << " seconds" << std::endl;
    };

    std::size_t row_index = 0;
    ChunkPlan current_plan = plan_chunk(row_index);
    if (current_plan.row_start >= current_plan.row_end) {
        return;
    }

    LoadedChunk current_chunk = load_chunk(current_plan);
    std::cout << "Data load cycle for " << current_chunk.points.size()
              << " points took " << current_chunk.load_seconds << " seconds" << std::endl;

    while (row_index < rows.size()) {
        auto chunk_cycle_start = std::chrono::steady_clock::now();

        std::size_t next_row_start = current_plan.row_end;

        std::unique_ptr<LoadedChunk> next_chunk;
        ChunkPlan next_plan;
        std::thread loader;
        if (next_row_start < rows.size()) {
            next_plan = plan_chunk(next_row_start);
            if (next_plan.row_start < next_plan.row_end) {
                next_chunk = std::make_unique<LoadedChunk>();
                loader = std::thread([&, next_plan, ptr = next_chunk.get()]() {
                    *ptr = load_chunk(next_plan);
                });
            }
        }

        std::size_t chunk_point_start = row_point_prefix[current_plan.row_start];
        std::size_t chunk_point_end = row_point_prefix[current_plan.row_end];

        process_chunk(current_chunk, chunk_point_start, chunk_point_end, chunk_cycle_start);

        if (loader.joinable()) {
            loader.join();
            std::cout << "Data load cycle for " << next_chunk->points.size()
                      << " points took " << next_chunk->load_seconds << " seconds" << std::endl;
            current_chunk = std::move(*next_chunk);
            current_plan = next_plan;
            row_index = current_plan.row_start;
        } else {
            break;
        }

        if (current_plan.row_start >= current_plan.row_end) {
            break;
        }
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
        std::cerr << "error to write " << filename << " file.\n";
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