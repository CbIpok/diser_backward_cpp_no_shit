#include "statistics.h"
#include "approx_orto.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <future>
#include <algorithm>
#include <limits>
#include "json.hpp"  // ����������� ���������� nlohmann::json

using json = nlohmann::json;

//// ���������� ��������������� ������������� � �������������� QR-����������
//std::pair<Eigen::VectorXd, Eigen::VectorXd> approximate_with_non_orthogonal_basis(const Eigen::VectorXd& x, const Eigen::MatrixXd& basis) {
//    // ������ ������� ������� ���������� ���������:
//    Eigen::VectorXd coeffs = basis.transpose().colPivHouseholderQr().solve(x);
//    // ��������� ������������� (�� ������������ �����)
//    Eigen::VectorXd approximation = basis.transpose() * coeffs;
//    return { approximation, coeffs };
//}

//
// ������� calculate_statistics
// ��������� ������ �� NetCDF (WaveManager � BasisManager), ����� ��� ������� ������� �������
// ��������� ������ ������� (wave_vector) � ��������������� ����� (smoothed_basis) � ���������
// ������������ ������������� ������� ������� (non orto) � ������� � ���������������� (orto).
//

int count_from_name(const std::string& name) {
    // ������� ������� ������� '_'
    std::size_t underscorePos = name.find('_');
    if (underscorePos != std::string::npos) {
        // ��������� ��������� ����� '_'
        std::string numberPart = name.substr(underscorePos + 1);
        // ����������� � ����� �����
        return std::stoi(numberPart);
    }
    // ���� '_' �� ������, ������ 0 ��� ����� ������ �������� �� ���������
    return 0;
}

void calculate_statistics(const std::string& root_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config,
    CoeffMatrix& statistics_orto,
    std::vector<int>& y_indices) {
    // ������������ �����
    std::string basis_path = root_folder + "/" + bath + "/" + basis;
    std::string wave_nc_path = root_folder + "/" + bath + "/" + wave + ".nc";

    BasisManager basis_manager(basis_path);
    WaveManager wave_manager(wave_nc_path);

    int width = area_config.all.empty() ? 0 : area_config.all[0];
    int height = area_config.all.size() > 1 ? area_config.all[1] : 0;

    int min_y = 0;
    int max_y = height;
    if (!area_config.mariogramm_bounds.empty()) {
        if (area_config.mariogramm_bounds.size() >= 4) {
            min_y = area_config.mariogramm_bounds[2];
            max_y = area_config.mariogramm_bounds[3];
        }
        else if (area_config.mariogramm_bounds.size() >= 2) {
            min_y = area_config.mariogramm_bounds[0];
            max_y = area_config.mariogramm_bounds[1];
        }
    }
    auto [time_dim, total_height, total_width] = wave_manager.get_dimensions();
    if (total_height > 0) {
        height = static_cast<int>(total_height);
    }
    if (total_width > 0 && width == 0) {
        width = static_cast<int>(total_width);
    }
    int max_min_y = height > 0 ? height - 1 : 0;
    min_y = std::clamp(min_y, 0, max_min_y);
    max_y = std::clamp(max_y, 0, height);
    if (max_y <= min_y) {
        max_y = std::min(height, min_y + 1);
    }
    std::size_t basis_count = basis_manager.basis_count();
    if (basis_count == 0) {
        int parsed_basis = count_from_name(basis);
        basis_count = static_cast<std::size_t>(parsed_basis > 0 ? parsed_basis : 1);
    }
    std::size_t time_steps = time_dim > 0 ? time_dim : 1;
    std::size_t width_dim = total_width > 0 ? total_width : static_cast<std::size_t>(width > 0 ? width : 1);
    std::size_t rows_available = static_cast<std::size_t>(std::max(1, max_y - min_y));
    const std::size_t max_memory_bytes = static_cast<std::size_t>(45ULL) * 1024 * 1024 * 1024;
    std::size_t bytes_per_row = time_steps * width_dim * sizeof(double) * (basis_count + 1);
    if (bytes_per_row == 0) {
        bytes_per_row = 1;
    }
    std::size_t rows_limit = std::max<std::size_t>(1, max_memory_bytes / bytes_per_row);
    std::size_t batch_size_sz = std::max<std::size_t>(1, std::min(rows_limit, rows_available));
    int batch_size = static_cast<int>(batch_size_sz);
    int y_start_init = min_y;

    statistics_orto.clear();
    y_indices.clear();

    for (int y_start = y_start_init; y_start < max_y; y_start += batch_size) {
        int y_end = std::min(y_start + batch_size, max_y);
        auto wave_data = wave_manager.load_mariogramm_by_region(y_start, y_end);
        auto fk_data = basis_manager.get_fk_region(y_start, y_end);
        if (wave_data.empty() || fk_data.empty()) continue;
        int T = wave_data.size();
        int region_height = wave_data[0].size();
        int n_basis = fk_data.size();

        int x_max = width / 4;
        std::cout << "loaded\n";

        std::vector<std::future<std::vector<CoefficientData>>> futures;
        futures.reserve(region_height);

        for (int i = 0; i < region_height; i++) {
            futures.push_back(std::async(std::launch::async, [i, T, x_max, n_basis, &wave_data, &fk_data]() -> std::vector<CoefficientData> {
                std::vector<CoefficientData> row_data;
                for (int x = 0; x < x_max; x++) {
                    // ������ ������� �������� ������� ��� �������� �������
                    Eigen::VectorXd wave_vector(T);
                    for (int t = 0; t < T; t++) {
                        wave_vector[t] = wave_data[t][i][x];
                    }
                    // ������ ������� ������ (n_basis x T)
                    Eigen::MatrixXd smoothed_basis(n_basis, T);
                    for (int b = 0; b < n_basis; b++) {
                        for (int t = 0; t < T; t++) {
                            smoothed_basis(b, t) = fk_data[b][t][i][x];
                        }
                    }
                    if (smoothed_basis.cols() != wave_vector.size()) continue;

                    CoefficientData pixelData;
                    try {
                        // ���������� ������������� ������������� ������� � ����������������
                        Eigen::VectorXd coefs_orto = approximate_with_non_orthogonal_basis_orto(wave_vector, smoothed_basis);
                        // ���������� ������������������� �������:
                        Eigen::VectorXd approximation = smoothed_basis.transpose() * coefs_orto;
                        // ���������� ������������������ ������ (RMSE)
                        double error = std::sqrt((wave_vector - approximation).squaredNorm() / wave_vector.size());
                        pixelData.coefs = coefs_orto;
                        pixelData.aprox_error = error;
                    } catch (const std::runtime_error& ex) {
                        std::string message = ex.what();
                        constexpr const char* zero_division_utf8 = u8"Деление на ноль: нормированный вектор слишком близок к нулю.";
                        constexpr const char zero_division_cp1251[] = "\xC4\xE5\xEB\xE5\xED\xE8\xE5 \xED\xE0 \xED\xEE\xEB\xFC: \xED\xEE\xF0\xEC\xE8\xF0\xEE\xE2\xE0\xED\xED\xFB\xE9 \xE2\xE5\xEA\xF2\xEE\xF0 \xF1\xEB\xE8\xF8\xEA\xEE\xEC \xE1\xEB\xE8\xE7\xEE\xEA \xEA \xED\xF3\xEB\xFE.";
                        const bool is_zero_division = message.find(zero_division_utf8) != std::string::npos ||
                            message.find(zero_division_cp1251) != std::string::npos;
                        if (is_zero_division) {
                            Eigen::VectorXd nan_vec = Eigen::VectorXd::Constant(n_basis, std::numeric_limits<double>::quiet_NaN());
                            pixelData.coefs = nan_vec;
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

        for (std::size_t i = 0; i < futures.size(); ++i) {
            auto row_data = futures[i].get();
            if (!row_data.empty()) {
                statistics_orto.push_back(row_data);
                y_indices.push_back(y_start + static_cast<int>(i));
            }
        }
    }
}

//// ������� ���������� ���������� ������� ������������� � CSV-����
//void save_coefficients_csv(const std::string& filename, const CoeffMatrix& coeffs) {
//    std::ofstream ofs(filename);
//    if (!ofs.is_open()) {
//        std::cerr << "�� ������� ������� ���� " << filename << " ��� ������.\n";
//        return;
//    }
//    // ������ ������ � ���� ��� ����������, ������ ������ �������� ������ ������������� (����� ��������� ���������)
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
//    std::cout << "���������: " << filename << "\n";
//}

void save_coefficients_json(const std::string& filename, const CoeffMatrix& coeffs, const std::vector<int>& y_indices) {
    nlohmann::json j;
    for (size_t row = 0; row < coeffs.size(); ++row) {
        for (size_t col = 0; col < coeffs[row].size(); ++col) {
            int y_value = (row < y_indices.size()) ? y_indices[row] : static_cast<int>(row);
            std::string key = "[" + std::to_string(y_value) + "," + std::to_string(col) + "]";
            // �������������� Eigen::VectorXd � std::vector<double>
            std::vector<double> vec(coeffs[row][col].coefs.data(),
                coeffs[row][col].coefs.data() + coeffs[row][col].coefs.size());
            double error = coeffs[row][col].aprox_error;
            j[key] = { {"coefs", vec}, {"aprox_error", error} };
        }
    }
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        std::cerr << "�� ������� ������� ���� " << filename << " ��� ������.\n";
        return;
    }
    ofs << j.dump(4);
    ofs.close();
    std::cout << "���������: " << filename << "\n";
}

// ������� save_and_plot_statistics: ��������� ���������� � ��������� ������������ � CSV
void save_and_plot_statistics(const std::string& root_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config) {
    CoeffMatrix statistics_orto;
    std::vector<int> y_indices;
    calculate_statistics(root_folder, bath, wave, basis, area_config, statistics_orto, y_indices);

    std::string filename_orto = "case_statistics_hd_y_" + basis + bath + "_o.json";
    //std::string filename_non_orto = "case_statistics_hd_y_" + basis + "_no.csv";

    save_coefficients_json(filename_orto, statistics_orto, y_indices);
    /*save_coefficients_json(filename_non_orto, statistics_non_orto);*/

    // ������������ �� ����������� � ������������ ����� ������� � Excel ��� �������� � Python ��� ���������� ��������.
}