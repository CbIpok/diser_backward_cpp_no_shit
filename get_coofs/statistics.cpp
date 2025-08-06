#define NOMINMAX 
#include "statistics.h"
#include "approx_orto.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <future>
#include <algorithm>
#include "json.hpp"

using json = nlohmann::json;

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

    std::string basis_path = root_folder + "/" + bath + "/" + basis;
    std::string wave_nc_path = root_folder + "/" + bath + "/" + wave + ".nc";

    BasisManager basis_manager(basis_path);
    WaveManager wave_manager(wave_nc_path);

    int width = area_config.all[0];
    int height = area_config.all[1];
    int gigabyte_size = 96;
    int memory_in_gb = 45;
    int batch_size = gigabyte_size * memory_in_gb / (count_from_name(basis) + 1);
    int y_start_init = 75;

    statistics_orto.clear();

    for (int y_start = y_start_init; y_start < height / 4; y_start += batch_size) {
        int y_end = std::min(y_start + batch_size, height);
        // Загружаем данные в виде плоских векторов.
        std::vector<double> wave_data = wave_manager.load_mariogramm_by_region(y_start, y_end);
        std::vector<std::vector<double>> fk_data = basis_manager.get_fk_region(y_start, y_end);
        if (wave_data.empty() || fk_data.empty())
            continue;

        int region_height = std::min(y_end, height / 4) - y_start;
        // Предполагаем, что x_max = width / 4 (как в оригинале)
        int x_max = width / 4;
        int totalElements = static_cast<int>(wave_data.size());
        // Вычисляем T, зная, что totalElements = T * region_height * x_max.
        int T = totalElements / (region_height * x_max);
        int n_basis = static_cast<int>(fk_data.size());

        std::cout << "loaded region: y_start=" << y_start << ", y_end=" << y_end << "\n";

        std::vector<std::future<std::vector<CoefficientData>>> futures;
        futures.reserve(region_height);

        for (int i = 0; i < region_height; i++) {
            futures.push_back(std::async(std::launch::async,
                [i, T, x_max, n_basis, &wave_data, &fk_data]() -> std::vector<CoefficientData> {
                    std::vector<CoefficientData> row_data;
                    // Данные организованы как [region_height][x_max][T]
                    for (int x = 0; x < x_max; x++) {
                        Eigen::VectorXd wave_vector(T);
                        for (int t = 0; t < T; t++) {
                            int idx = i * (x_max * T) + x * T + t;
                            wave_vector[t] = wave_data[idx];
                        }
                        Eigen::MatrixXd smoothed_basis(n_basis, T);
                        for (int b = 0; b < n_basis; b++) {
                            for (int t = 0; t < T; t++) {
                                int idx = i * (x_max * T) + x * T + t;
                                smoothed_basis(b, t) = fk_data[b][idx];
                            }
                        }
                        if (smoothed_basis.cols() != wave_vector.size())
                            continue;
                        Eigen::VectorXd coefs_orto = approximate_with_non_orthogonal_basis_orto(wave_vector, smoothed_basis);
                        Eigen::VectorXd approximation = smoothed_basis.transpose() * coefs_orto;
                        double error = std::sqrt((wave_vector - approximation).squaredNorm() / wave_vector.size());

                        CoefficientData pixelData;
                        pixelData.coefs = coefs_orto;
                        pixelData.aprox_error = error;
                        row_data.push_back(pixelData);
                    }
                    return row_data;
                }
            ));
        }

        for (auto& future : futures) {
            auto row_data = future.get();
            if (!row_data.empty())
                statistics_orto.push_back(row_data);
        }
    }
}

void save_coefficients_json(const std::string& filename, const CoeffMatrix& coeffs) {
    nlohmann::json j;
    for (size_t row = 0; row < coeffs.size(); ++row) {
        for (size_t col = 0; col < coeffs[row].size(); ++col) {
            std::string key = "[" + std::to_string(row) + "," + std::to_string(col) + "]";
            std::vector<double> vec(coeffs[row][col].coefs.data(),
                coeffs[row][col].coefs.data() + coeffs[row][col].coefs.size());
            double error = coeffs[row][col].aprox_error;
            j[key] = { {"coefs", vec}, {"aprox_error", error} };
        }
    }
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "Cannot open file " << filename << " for writing.\n";
        return;
    }
    ofs << j.dump(4);
    ofs.close();
    std::cout << "Saved: " << filename << "\n";
}

void save_and_plot_statistics(const std::string& root_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config) {
    CoeffMatrix statistics_orto;
    calculate_statistics(root_folder, bath, wave, basis, area_config, statistics_orto);

    std::string filename_orto = "E:/tsunami_res_dir/coefs_nessesary/case_statistics_hd_y_" + basis + "_" + bath + "_right.json";
    save_coefficients_json(filename_orto, statistics_orto);
}
