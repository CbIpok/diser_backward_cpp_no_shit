// statistics.cpp
#include "statistics.h"
#include "approx_orto.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <future>
#include <algorithm>
#include <utility>
#include <thread>
#include <iterator>
#include <chrono>
#include "json.hpp"
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <string_view>

using json = nlohmann::json;
namespace bg = boost::geometry;
using Point2i = bg::model::d2::point_xy<int>;
using Clock = std::chrono::steady_clock;

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

    int minY = area_config.height;
    int maxY = 0;
    int minX = area_config.width;
    int maxX = 0;
    for (auto const& p : area_config.mariogramm_poly.outer()) {
        minX = std::min(minX, int(bg::get<0>(p)));
        maxX = std::max(maxX, int(bg::get<0>(p)));
        minY = std::min(minY, int(bg::get<1>(p)));
        maxY = std::max(maxY, int(bg::get<1>(p)));
    }

    BasisManager basis_manager(root_folder + "/" + bath + "/" + basis);
    WaveManager wave_manager(root_folder + "/" + bath + "/" + wave + ".nc");

    const int H = std::max(0, maxY - minY);
    if (H == 0) {
        return;
    }

    size_t wave_T = 0, wave_Y = 0, wave_X = 0;
    if (!wave_manager.describe(wave_T, wave_Y, wave_X)) {
        std::cerr << "[ERROR] Unable to read wave dimensions" << std::endl;
        return;
    }

    size_t basis_count = 0, basis_T = 0, basis_Y = 0, basis_X = 0;
    if (!basis_manager.describe(basis_count, basis_T, basis_Y, basis_X)) {
        std::cerr << "[ERROR] Unable to read basis dimensions" << std::endl;
        return;
    }

    const int n_basis = static_cast<int>(basis_count);
    const int T = static_cast<int>(wave_T);
    const int W = static_cast<int>(wave_X);

    if (n_basis <= 0 || T <= 0 || W <= 0) {
        return;
    }

    constexpr std::size_t MAX_MEMORY_BYTES = 45ULL * 1024ULL * 1024ULL * 1024ULL; // 45 GB
    std::size_t bytes_per_row = std::max<std::size_t>(1, static_cast<std::size_t>(n_basis + 1) * wave_T * wave_X * sizeof(double));
    int band_height = static_cast<int>(std::max<std::size_t>(1, MAX_MEMORY_BYTES / bytes_per_row));
    band_height = std::max(1, std::min(band_height, H));

    statistics_orto.clear();
    const unsigned hardware_threads = std::max(1u, std::thread::hardware_concurrency());

    for (int band_start = 0; band_start < H; band_start += band_height) {
        const int band_end = std::min(H, band_start + band_height);
        const int global_start = minY + band_start;
        const int global_end = minY + band_end;

        const auto load_start = Clock::now();
        auto fk_data = basis_manager.get_fk_region(global_start, global_end);
        auto wave_data = wave_manager.load_mariogramm_by_region(global_start, global_end);
        const auto load_end = Clock::now();
        const auto load_ms = std::chrono::duration_cast<std::chrono::milliseconds>(load_end - load_start).count();
        std::cout << "[TIMING] Batch rows [" << global_start << "," << global_end << ") load took " << load_ms << " ms" << std::endl;

        if (fk_data.empty() || wave_data.empty()) {
            continue;
        }

        const int bandH = band_end - band_start;
        if (bandH <= 0) {
            continue;
        }

        const auto compute_start = Clock::now();

        int rows_per_thread = std::max(1, (bandH + static_cast<int>(hardware_threads) - 1) / static_cast<int>(hardware_threads));
        std::vector<std::pair<int, int>> row_ranges;
        row_ranges.reserve(static_cast<std::size_t>((bandH + rows_per_thread - 1) / rows_per_thread));
        for (int start = 0; start < bandH; start += rows_per_thread) {
            const int end = std::min(start + rows_per_thread, bandH);
            row_ranges.emplace_back(start, end);
        }

        const int total_width = std::max(1, maxX - minX + 1);
        const int default_tile_width = 128;
        const int min_preferred_tile_width = 64;
        int tile_width = std::min(default_tile_width, total_width);
        if (tile_width < min_preferred_tile_width && total_width >= min_preferred_tile_width) {
            tile_width = min_preferred_tile_width;
        }
        std::size_t col_tiles = static_cast<std::size_t>((total_width + tile_width - 1) / tile_width);
        if (col_tiles == 0) {
            col_tiles = 1;
            tile_width = total_width;
        }
        while (!row_ranges.empty() && row_ranges.size() * col_tiles < hardware_threads && tile_width > 1) {
            int next_width = tile_width > min_preferred_tile_width
                ? std::max(min_preferred_tile_width, tile_width / 2)
                : std::max(tile_width / 2, 1);
            if (next_width == tile_width) {
                if (tile_width == 1) {
                    break;
                }
                next_width = 1;
            }
            tile_width = next_width;
            col_tiles = static_cast<std::size_t>((total_width + tile_width - 1) / tile_width);
        }

        std::vector<std::pair<int, int>> col_ranges;
        col_ranges.reserve(col_tiles);
        for (int start = minX; start <= maxX; start += tile_width) {
            const int end = std::min(start + tile_width, maxX + 1);
            if (start < end) {
                col_ranges.emplace_back(start, end);
            }
        }
        if (col_ranges.empty()) {
            col_ranges.emplace_back(minX, maxX + 1);
        }

        std::vector<std::vector<CoeffMatrix>> tile_results(row_ranges.size(), std::vector<CoeffMatrix>(col_ranges.size()));
        std::vector<std::future<void>> futures;

        for (std::size_t row_idx = 0; row_idx < row_ranges.size(); ++row_idx) {
            const int row_start = row_ranges[row_idx].first;
            const int row_end = row_ranges[row_idx].second;
            for (std::size_t col_idx = 0; col_idx < col_ranges.size(); ++col_idx) {
                const int col_start = col_ranges[col_idx].first;
                const int col_end = col_ranges[col_idx].second;
                futures.emplace_back(std::async(std::launch::async, [&, row_idx, col_idx, row_start, row_end, col_start, col_end]() {
                    CoeffMatrix local(row_end - row_start);
                    for (int i = row_start; i < row_end; ++i) {
                        std::vector<CoefficientData> row;
                        row.reserve(static_cast<std::size_t>(std::max(0, col_end - col_start)));
                        for (int x = col_start; x < col_end; ++x) {
                            const Point2i pt{ x, minY + band_start + i };
                            if (bg::within(pt, area_config.mariogramm_poly)) {
                                Eigen::VectorXd wave_vec(T);
                                for (int t = 0; t < T; ++t) {
                                    wave_vec[t] = wave_data[t][i][x];
                                }

                                Eigen::MatrixXd B(n_basis, T);
                                for (int b = 0; b < n_basis; ++b) {
                                    for (int t = 0; t < T; ++t) {
                                        B(b, t) = fk_data[b][t][i][x];
                                    }
                                }

                                try {
                                    Eigen::VectorXd coefs = approximate_with_non_orthogonal_basis_orto(wave_vec, B);
                                    Eigen::VectorXd approx = B.transpose() * coefs;
                                    double err = std::sqrt((wave_vec - approx).squaredNorm() / T);

                                    CoefficientData data;
                                    data.pt = pt;
                                    data.coefs = std::move(coefs);
                                    data.aprox_error = err;
                                    row.push_back(std::move(data));
                                }
                                catch (const std::runtime_error& ex) {
                                    if (std::string_view(ex.what()).find("Division by zero") != std::string_view::npos) {
                                        CoefficientData data;
                                        data.pt = pt;
                                        data.is_nan = true;
                                        row.push_back(std::move(data));
                                    }
                                    else {
                                        throw;
                                    }
                                }
                            }
                        }
                        local[i - row_start] = std::move(row);
                    }
                    tile_results[row_idx][col_idx] = std::move(local);
                }));
            }
        }

        for (auto& f : futures) {
            f.get();
        }

        std::vector<std::vector<CoefficientData>> band_results(bandH);
        for (std::size_t row_idx = 0; row_idx < row_ranges.size(); ++row_idx) {
            const int row_start = row_ranges[row_idx].first;
            const int row_end = row_ranges[row_idx].second;
            for (int i = row_start; i < row_end; ++i) {
                const std::size_t local_idx = static_cast<std::size_t>(i - row_start);
                std::size_t total_row_size = 0;
                for (std::size_t col_idx = 0; col_idx < col_ranges.size(); ++col_idx) {
                    const auto& tile_row = tile_results[row_idx][col_idx];
                    if (local_idx < tile_row.size()) {
                        total_row_size += tile_row[local_idx].size();
                    }
                }
                if (total_row_size == 0) {
                    continue;
                }

                auto& dst_row = band_results[i];
                dst_row.reserve(total_row_size);
                for (std::size_t col_idx = 0; col_idx < col_ranges.size(); ++col_idx) {
                    auto& tile_row = tile_results[row_idx][col_idx];
                    if (local_idx >= tile_row.size()) {
                        continue;
                    }
                    auto& segment = tile_row[local_idx];
                    if (!segment.empty()) {
                        std::move(segment.begin(), segment.end(), std::back_inserter(dst_row));
                        segment.clear();
                    }
                }
            }
        }

        for (auto& row : band_results) {
            if (!row.empty()) {
                statistics_orto.push_back(std::move(row));
            }
        }

        const auto compute_end = Clock::now();
        const auto compute_ms = std::chrono::duration_cast<std::chrono::milliseconds>(compute_end - compute_start).count();
        std::cout << "[TIMING] Batch rows [" << global_start << "," << global_end << ") compute took " << compute_ms << " ms" << std::endl;
    }
}

void save_coefficients_json(const std::string& filename, const CoeffMatrix& coeffs) {
    nlohmann::json j;
    for (size_t row = 0; row < coeffs.size(); ++row) {
        for (size_t col = 0; col < coeffs[row].size(); ++col) {
            std::string key = "[" + std::to_string(bg::get<0>(coeffs[row][col].pt)) + "," + std::to_string(bg::get<1>(coeffs[row][col].pt)) + "]";
            if (coeffs[row][col].is_nan) {
                j[key] = "nan";
                continue;
            }

            std::vector<double> vec(coeffs[row][col].coefs.data(),
                coeffs[row][col].coefs.data() + coeffs[row][col].coefs.size());
            double error = coeffs[row][col].aprox_error;
            j[key] = { {"coefs", vec}, {"aprox_error", error} };
        }
    }
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "error to write " << filename << " file." << std::endl;
        return;
    }
    ofs << j.dump(4);
    ofs.close();
    std::cout << "saved: " << filename << std::endl;
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
