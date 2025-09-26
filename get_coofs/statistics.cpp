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

    int H = maxY - minY;

    // Для оценки ширины полосы чтения загружаем одну строку
    auto fk_sample = basis_manager.get_fk_region(minY, minY + 1);
    auto wave_sample = wave_manager.load_mariogramm_by_region(minY, minY + 1);
    if (fk_sample.empty() || wave_sample.empty()) return;

    int T = static_cast<int>(wave_sample.size());
    int W = static_cast<int>(wave_sample[0][0].size());
    int n_basis = static_cast<int>(fk_sample.size());

    constexpr std::size_t MAX_MEMORY_BYTES = 45ULL * 1024ULL * 1024ULL * 1024ULL; // 50 GB
    std::size_t bytes_per_row = static_cast<std::size_t>(n_basis + 1) * T * W * sizeof(double);
    int band_height = static_cast<int>(std::max<std::size_t>(1, MAX_MEMORY_BYTES / bytes_per_row));

    for (int band_start = 0; band_start < H; band_start += band_height) {
        int band_end = std::min(H, band_start + band_height);

        auto fk_data = basis_manager.get_fk_region(minY + band_start, minY + band_end);
        auto wave_data = wave_manager.load_mariogramm_by_region(minY + band_start, minY + band_end);
        if (fk_data.empty() || wave_data.empty()) continue;

        int bandH = band_end - band_start;
        if (bandH <= 0)
            continue;

        // 2) параллельно по строкам блока на 24 потока
        constexpr int THREADS = 48;
        int rows_per_thread = std::max(1, (bandH + THREADS - 1) / THREADS);

        // заранее режем загруженный блок на жирные диапазоны строк,
        // чтобы каждый поток получил свою крупную порцию работы
        std::vector<std::pair<int, int>> row_ranges;
        row_ranges.reserve((bandH + rows_per_thread - 1) / rows_per_thread);
        for (int start = 0; start < bandH; start += rows_per_thread) {
            int end = std::min(start + rows_per_thread, bandH);
            row_ranges.emplace_back(start, end);
        }

        const int total_width = std::max(1, maxX - minX + 1);
        const unsigned hardware_threads = std::max(1u, std::thread::hardware_concurrency());
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
            int end = std::min(start + tile_width, maxX + 1);
            if (start < end) {
                col_ranges.emplace_back(start, end);
            }
        }
        if (col_ranges.empty()) {
            col_ranges.emplace_back(minX, maxX + 1);
        }

        CoeffMatrix band_results(bandH);
        const int band_width = total_width;
        std::vector<std::vector<char>> inside_mask(static_cast<std::size_t>(bandH), std::vector<char>(static_cast<std::size_t>(band_width), 0));
        std::vector<std::vector<int>> prefix_indices(static_cast<std::size_t>(bandH), std::vector<int>(static_cast<std::size_t>(band_width + 1), 0));

        for (int row = 0; row < bandH; ++row) {
            auto& mask_row = inside_mask[static_cast<std::size_t>(row)];
            auto& prefix_row = prefix_indices[static_cast<std::size_t>(row)];
            prefix_row[0] = 0;
            for (int col = 0; col < band_width; ++col) {
                Point2i pt{ minX + col, minY + band_start + row };
                bool inside = bg::within(pt, area_config.mariogramm_poly);
                mask_row[static_cast<std::size_t>(col)] = static_cast<char>(inside);
                prefix_row[static_cast<std::size_t>(col + 1)] = prefix_row[static_cast<std::size_t>(col)] + (inside ? 1 : 0);
            }
            band_results[static_cast<std::size_t>(row)].resize(static_cast<std::size_t>(prefix_row[static_cast<std::size_t>(band_width)]));
        }
        std::vector<std::future<void>> futs;
        futs.reserve(row_ranges.size() * col_ranges.size());

        for (std::size_t row_idx = 0; row_idx < row_ranges.size(); ++row_idx) {
            int row_start = row_ranges[row_idx].first;
            int row_end = row_ranges[row_idx].second;
            if (row_start >= row_end) {
                continue;
            }
            for (std::size_t col_idx = 0; col_idx < col_ranges.size(); ++col_idx) {
                int col_start = col_ranges[col_idx].first;
                int col_end = col_ranges[col_idx].second;
                if (col_start >= col_end) {
                    continue;
                }
                futs.emplace_back(std::async(std::launch::async, [&, row_start, row_end, col_start, col_end]() {
                    for (int i = row_start; i < row_end; ++i) {
                        auto& mask_row = inside_mask[static_cast<std::size_t>(i)];
                        auto& prefix_row = prefix_indices[static_cast<std::size_t>(i)];
                        auto& dst_row = band_results[static_cast<std::size_t>(i)];
                        for (int x = col_start; x < col_end; ++x) {
                            int col_local = x - minX;
                            if (col_local < 0 || col_local >= band_width) {
                                continue;
                            }

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

                            auto coefs = approximate_with_non_orthogonal_basis_orto(wave_vec, B);
                            Eigen::VectorXd approx = B.transpose() * coefs;
                            double err = std::sqrt((wave_vec - approx).squaredNorm() / T);

                            if (mask_row[static_cast<std::size_t>(col_local)]) {
                                std::size_t insert_idx = static_cast<std::size_t>(prefix_row[static_cast<std::size_t>(col_local)]);
                                if (insert_idx < dst_row.size()) {
                                    dst_row[insert_idx] = { Point2i{ x, minY + band_start + i }, std::move(coefs), err };
                                }
                            }
                        }
                    }
                }));
            }
        }

        for (auto& f : futs) {
            f.get();
        }

        for (auto& row : band_results) {
            if (!row.empty())
                statistics_orto.push_back(std::move(row));
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