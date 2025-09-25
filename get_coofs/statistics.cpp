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
    WaveManager  wave_manager(root_folder + "/" + bath + "/" + wave + ".nc");

    std::size_t wave_T = 0, wave_Y = 0, wave_X = 0;
    if (!wave_manager.get_dimensions(wave_T, wave_Y, wave_X)) {
        std::cerr << "failed to query wave dimensions for " << wave_manager.nc_file << '\n';
        return;
    }

    std::size_t basis_T = 0, basis_Y = 0, basis_X = 0;
    if (!basis_manager.get_dimensions(basis_T, basis_Y, basis_X)) {
        std::cerr << "failed to query basis dimensions for " << basis_manager.folder << '\n';
        return;
    }

    if (basis_T != wave_T || basis_Y != wave_Y || basis_X != wave_X) {
        std::cerr << "basis dataset dimensions do not match wave dataset" << '\n';
        return;
    }

    std::size_t basis_count = basis_manager.basis_count();
    if (basis_count == 0) {
        std::cerr << "no basis files found in " << basis_manager.folder << '\n';
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

    std::vector<Point2i> points;
    points.reserve(static_cast<std::size_t>(maxY - minY + 1)
        * static_cast<std::size_t>(maxX - minX + 1));

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            Point2i pt(x, y);
            if (bg::within(pt, area_config.mariogramm_poly)) {
                points.emplace_back(pt);
            }
        }
    }

    if (points.empty()) {
        return;
    }

    constexpr std::size_t MAX_MEMORY_BYTES = 45ULL * 1024ULL * 1024ULL * 1024ULL; // 45 GB
    std::size_t bytes_per_point = static_cast<std::size_t>(n_basis + 1)
        * static_cast<std::size_t>(T) * sizeof(double);

    std::size_t max_points_by_memory = bytes_per_point > 0 ? MAX_MEMORY_BYTES / bytes_per_point : 0;
    if (max_points_by_memory == 0) {
        max_points_by_memory = 1;
    }

    std::size_t points_per_chunk = max_points_by_memory;
    if (points_per_chunk > 1) {
        // Double buffering keeps one chunk in compute while the next loads, so
        // halve the memory target to honour the overall limit.
        points_per_chunk = std::max<std::size_t>(1, points_per_chunk / 2);
    }

    std::size_t total_points = points.size();

    struct LoadedChunk {
        std::vector<Point2i> points;
        std::vector<std::vector<double>> wave;
        std::vector<std::vector<std::vector<double>>> fk;
        double load_seconds = 0.0;
    };

    auto load_chunk = [&](std::size_t start, std::size_t end) {
        LoadedChunk chunk;
        chunk.points.assign(points.begin() + start, points.begin() + end);
        auto load_start = std::chrono::steady_clock::now();
        chunk.wave = wave_manager.load_mariogramm_points(chunk.points, T, data_height, W);
        chunk.fk = basis_manager.get_fk_points(chunk.points, T, data_height, W);
        auto load_end = std::chrono::steady_clock::now();
        chunk.load_seconds = std::chrono::duration<double>(load_end - load_start).count();
        return chunk;
    };

    auto process_chunk = [&](const LoadedChunk& chunk,
        std::size_t chunk_start,
        std::size_t chunk_end,
        const std::chrono::steady_clock::time_point& chunk_cycle_start) {
        if (chunk.wave.size() != chunk.points.size()) {
            return;
        }

        bool fk_valid = chunk.fk.size() == static_cast<std::size_t>(n_basis);
        if (fk_valid) {
            for (const auto& basis_points : chunk.fk) {
                if (basis_points.size() != chunk.points.size()) {
                    fk_valid = false;
                    break;
                }
            }
        }
        if (!fk_valid) {
            return;
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
                    const auto& wave_series = chunk.wave[idx];
                    if (wave_series.size() != static_cast<std::size_t>(T)) {
                        continue;
                    }

                    bool basis_ok = true;
                    for (int b = 0; b < n_basis; ++b) {
                        const auto& basis_series = chunk.fk[static_cast<std::size_t>(b)][idx];
                        if (basis_series.size() != static_cast<std::size_t>(T)) {
                            basis_ok = false;
                            break;
                        }
                        for (int t = 0; t < T; ++t) {
                            B(b, t) = basis_series[static_cast<std::size_t>(t)];
                        }
                    }
                    if (!basis_ok) {
                        continue;
                    }

                    for (int t = 0; t < T; ++t) {
                        wave_vec[t] = wave_series[static_cast<std::size_t>(t)];
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

    std::size_t chunk_start = 0;
    std::size_t chunk_end = std::min(total_points, chunk_start + points_per_chunk);
    LoadedChunk current_chunk = load_chunk(chunk_start, chunk_end);
    std::cout << "Data load cycle for " << current_chunk.points.size()
              << " points took " << current_chunk.load_seconds << " seconds" << std::endl;

    while (chunk_start < total_points) {
        auto chunk_cycle_start = std::chrono::steady_clock::now();

        std::size_t next_start = chunk_end;
        std::size_t next_end = std::min(total_points, next_start + points_per_chunk);

        std::unique_ptr<LoadedChunk> next_chunk;
        std::thread loader;
        if (next_start < total_points) {
            next_chunk = std::make_unique<LoadedChunk>();
            loader = std::thread([&, next_start, next_end, ptr = next_chunk.get()]() {
                *ptr = load_chunk(next_start, next_end);
            });
        }

        process_chunk(current_chunk, chunk_start, chunk_end, chunk_cycle_start);

        if (loader.joinable()) {
            loader.join();
            std::cout << "Data load cycle for " << next_chunk->points.size()
                      << " points took " << next_chunk->load_seconds << " seconds" << std::endl;
            current_chunk = std::move(*next_chunk);
            chunk_start = next_start;
            chunk_end = next_end;
        } else {
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