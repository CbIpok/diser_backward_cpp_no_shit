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

        // 2) параллельно обрабатываем строки блока, деля их на шесть потоков
        CoeffMatrix band_results(bandH);
        constexpr int THREADS = 6;
        int actual_threads = std::min(THREADS, std::max(1, bandH));
        int rows_per_thread = (bandH + actual_threads - 1) / actual_threads;

        std::vector<std::future<void>> futs;
        futs.reserve(actual_threads);

        for (int thread_idx = 0; thread_idx < actual_threads; ++thread_idx) {
            int row_start = thread_idx * rows_per_thread;
            int row_end = std::min(bandH, row_start + rows_per_thread);
            if (row_start >= row_end) {
                continue;
            }

            futs.emplace_back(std::async(std::launch::async, [&, row_start, row_end]() {
                for (int i = row_start; i < row_end; ++i) {
                    std::vector<CoefficientData> row;
                    row.reserve(static_cast<std::size_t>(std::max(0, maxX - minX + 1)));
                    for (int x = minX; x <= maxX; ++x) {
                        if (x < 0 || x >= W) {
                            continue;
                        }

                        Point2i pt{ x, minY + band_start + i };
                        if (!bg::within(pt, area_config.mariogramm_poly))
                            continue;

                        Eigen::VectorXd wave_vec(T);
                        for (int t = 0; t < T; ++t)
                            wave_vec[t] = wave_data[t][i][x];

                        Eigen::MatrixXd B(n_basis, T);
                        for (int b = 0; b < n_basis; ++b)
                            for (int t = 0; t < T; ++t)
                                B(b, t) = fk_data[b][t][i][x];

                        auto coefs = approximate_with_non_orthogonal_basis_orto(wave_vec, B);
                        Eigen::VectorXd approx = B.transpose() * coefs;
                        double err = std::sqrt((wave_vec - approx).squaredNorm() / T);

                        row.push_back({ pt, coefs, err });
                    }
                    band_results[i] = std::move(row);
                }
            }));
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