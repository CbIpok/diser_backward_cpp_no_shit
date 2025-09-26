#include "statistics.h"
#include "approx_orto.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <future>
#include <algorithm>
#include <cmath>
#include <limits>
#include "json.hpp"  // пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ nlohmann::json

using json = nlohmann::json;

//// пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ QR-пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ
//std::pair<Eigen::VectorXd, Eigen::VectorXd> approximate_with_non_orthogonal_basis(const Eigen::VectorXd& x, const Eigen::MatrixXd& basis) {
//    // пїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ:
//    Eigen::VectorXd coeffs = basis.transpose().colPivHouseholderQr().solve(x);
//    // пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ (пїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅ)
//    Eigen::VectorXd approximation = basis.transpose() * coeffs;
//    return { approximation, coeffs };
//}

//
// пїЅпїЅпїЅпїЅпїЅпїЅпїЅ calculate_statistics
// пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅ NetCDF (WaveManager пїЅ BasisManager), пїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ
// пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ (wave_vector) пїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅ (smoothed_basis) пїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ
// пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ (non orto) пїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ (orto).
//

void calculate_statistics(const std::string& root_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config,
    CoeffMatrix& statistics_orto) {
    // пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅ
    std::string basis_path = root_folder + "/" + bath + "/" + basis;
    std::string wave_nc_path = root_folder + "/" + bath + "/" + wave + ".nc";

    BasisManager basis_manager(basis_path);
    WaveManager wave_manager(wave_nc_path);

    int width = area_config.all[0];
    int height = area_config.all[1];

    const std::size_t basis_files = basis_manager.basis_count();
    constexpr double memory_limit_bytes = 45.0 * 1024.0 * 1024.0 * 1024.0;

    std::size_t sample_T = 0;
    std::size_t sample_width = 0;
    int sample_region_end = std::min(height, 1);
    if (sample_region_end <= 0) {
        sample_region_end = 1;
    }
    auto sample_wave = wave_manager.load_mariogramm_by_region(0, sample_region_end);
    if (!sample_wave.empty()) {
        sample_T = sample_wave.size();
        if (!sample_wave[0].empty()) {
            sample_width = sample_wave[0][0].size();
        }
    }
    sample_wave.clear();

    int batch_size = 1;
    if (sample_T > 0 && sample_width > 0 && basis_files > 0) {
        double bytes_per_row = static_cast<double>(sample_T) * static_cast<double>(sample_width) * sizeof(double);
        double estimated_row_usage = bytes_per_row * (static_cast<double>(basis_files) + 1.0) * 2.0;
        if (estimated_row_usage > 0.0) {
            batch_size = static_cast<int>(memory_limit_bytes / estimated_row_usage);
        }
    }
    if (batch_size < 1) {
        batch_size = 1;
    }
    int y_start_init = 75;

    statistics_orto.clear();

    for (int y_start = y_start_init; y_start < height / 4; y_start += batch_size) {
        int y_end = std::min(y_start + batch_size, height);
        auto wave_data = wave_manager.load_mariogramm_by_region(y_start, y_end);
        auto fk_data = basis_manager.get_fk_region(y_start, y_end);
        if (wave_data.empty() || fk_data.empty()) continue;
        int T = wave_data.size();
        int region_height = wave_data[0].size();
        int region_width = wave_data[0][0].size();
        int n_basis = fk_data.size();

        int x_max = width / 4;
        std::cout << "loaded\n";

        std::vector<std::future<std::vector<CoefficientData>>> futures;
        futures.reserve(region_height);

        for (int i = 0; i < region_height; i++) {
            futures.push_back(std::async(std::launch::async, [i, T, x_max, n_basis, &wave_data, &fk_data]() -> std::vector<CoefficientData> {
                std::vector<CoefficientData> row_data;
                for (int x = 0; x < x_max; x++) {
                    // пїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ
                    Eigen::VectorXd wave_vector(T);
                    for (int t = 0; t < T; t++) {
                        wave_vector[t] = wave_data[t][i][x];
                    }
                    // пїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅ (n_basis x T)
                    Eigen::MatrixXd smoothed_basis(n_basis, T);
                    for (int b = 0; b < n_basis; b++) {
                        for (int t = 0; t < T; t++) {
                            smoothed_basis(b, t) = fk_data[b][t][i][x];
                        }
                    }
                    if (smoothed_basis.cols() != wave_vector.size()) continue;

                    // пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ
                    CoefficientData pixelData;
                    try {
                        Eigen::VectorXd coefs_orto = approximate_with_non_orthogonal_basis_orto(wave_vector, smoothed_basis);
                        Eigen::VectorXd approximation = smoothed_basis.transpose() * coefs_orto;
                        double error = std::sqrt((wave_vector - approximation).squaredNorm() / wave_vector.size());
                        pixelData.coefs = coefs_orto;
                        pixelData.aprox_error = error;
                    }
                    catch (const std::runtime_error& ex) {
                        if (std::string(ex.what()) == kZeroNormVectorError) {
                            pixelData.coefs = Eigen::VectorXd::Constant(n_basis, std::numeric_limits<double>::quiet_NaN());
                            pixelData.aprox_error = std::numeric_limits<double>::quiet_NaN();
                        }
                        else {
                            throw;
                        }
                    }
                    row_data.push_back(pixelData);
                }
                return row_data;
                }));
        }

        for (auto& future : futures) {
            auto row_data = future.get();
            if (!row_data.empty()) {
                statistics_orto.push_back(row_data);
            }
        }
    }
}

//// пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅ CSV-пїЅпїЅпїЅпїЅ
//void save_coefficients_csv(const std::string& filename, const CoeffMatrix& coeffs) {
//    std::ofstream ofs(filename);
//    if (!ofs.is_open()) {
//        std::cerr << "пїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅ " << filename << " пїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅ.\n";
//        return;
//    }
//    // пїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅ пїЅ пїЅпїЅпїЅпїЅ пїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ, пїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ (пїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ)
//    for (const auto& row : coeffs) {
//        bool firstCell = true;
//        for (const auto& vec : row) {
//            if (!firstCell) ofs << ",";
//            firstCell = false;
//            std::ostringstream oss;
//            for (int i = 0; i < vec.size(); i++) {
//                oss << vec[i];
//                if (i + 1 < vec.size()) oss << " ";
//            }
//            ofs << oss.str();
//        }
//        ofs << "\n";
//    }
//    ofs.close();
//    std::cout << "пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ: " << filename << "\n";
//}

void save_coefficients_json(const std::string& filename, const CoeffMatrix& coeffs, const AreaConfigurationInfo& area_config) {
    nlohmann::json j;
    for (size_t row = 0; row < coeffs.size(); ++row) {
        for (size_t col = 0; col < coeffs[row].size(); ++col) {
            std::string key = "[" + std::to_string(row) + "," + std::to_string(col) + "]";
            // пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ Eigen::VectorXd пїЅ std::vector<double>
            std::vector<double> vec(coeffs[row][col].coefs.data(),
                coeffs[row][col].coefs.data() + coeffs[row][col].coefs.size());
            double error = coeffs[row][col].aprox_error;
            j[key] = { {"coefs", vec}, {"aprox_error", error} };
        }
    }
    bool has_bounds = false;
    int minX = std::numeric_limits<int>::max();
    int maxX = std::numeric_limits<int>::lowest();
    int minY = std::numeric_limits<int>::max();
    int maxY = std::numeric_limits<int>::lowest();

    if (!area_config.mariogramm_poly.outer().empty()) {
        for (const auto& p : area_config.mariogramm_poly.outer()) {
            minX = std::min(minX, static_cast<int>(bg::get<0>(p)));
            maxX = std::max(maxX, static_cast<int>(bg::get<0>(p)));
            minY = std::min(minY, static_cast<int>(bg::get<1>(p)));
            maxY = std::max(maxY, static_cast<int>(bg::get<1>(p)));
        }
        has_bounds = minX <= maxX && minY <= maxY;
    }

    if (has_bounds) {
        j["_meta"] = {
            {"bounds", {
                {"min_x", minX},
                {"max_x", maxX},
                {"min_y", minY},
                {"max_y", maxY}
            }}
        };
    }

    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "пїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅ " << filename << " пїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅ.\n";
        return;
    }
    ofs << j.dump(4);
    ofs.close();
    std::cout << "пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ: " << filename << "\n";
}

// пїЅпїЅпїЅпїЅпїЅпїЅпїЅ save_and_plot_statistics: пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅ CSV
void save_and_plot_statistics(const std::string& root_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config) {
    CoeffMatrix statistics_orto;
    calculate_statistics(root_folder, bath, wave, basis, area_config, statistics_orto);

    std::string filename_orto = "case_statistics_hd_y_" + basis + bath + "_o.json";
    //std::string filename_non_orto = "case_statistics_hd_y_" + basis + "_no.csv";

    save_coefficients_json(filename_orto, statistics_orto, area_config);
    /*save_coefficients_json(filename_non_orto, statistics_non_orto);*/

    // пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅ Excel пїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅ Python пїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ пїЅпїЅпїЅпїЅпїЅпїЅпїЅпїЅ.
}
