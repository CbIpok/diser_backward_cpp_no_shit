// statistics.cpp
#include "statistics.h"
#include "approx_orto.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <future>
#include <algorithm>
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
    // прочитать весь прямоугольник [minY..maxY)
    BasisManager basis_manager(root_folder + "/" + bath + "/" + basis);
    WaveManager  wave_manager(root_folder + "/" + bath + "/" + wave + ".nc");

    auto fk_data = basis_manager.get_fk_region(minY, maxY);
    auto wave_data = wave_manager.load_mariogramm_by_region(minY, maxY);
    if (fk_data.empty() || wave_data.empty()) return;

    int T = int(wave_data.size());
    int H = maxY - minY;
    int W = area_config.width;
    int n_basis = int(fk_data.size());

    // 2) параллельно по строкам в прям-ке
    std::vector<std::future<std::vector<CoefficientData>>> futs;
    futs.reserve(H);
    for (int i = 0; i < H; ++i) {
        futs.push_back(std::async(std::launch::async,
            [&, i]() -> std::vector<CoefficientData> {
                std::vector<CoefficientData> row;
                for (int x = minX; x <= maxX; ++x) {
                    // глобальные координаты
                    Point2i pt{ x, minY + i };
                    // проверяем, внутри ли полигона
                    if (!bg::within(pt, area_config.mariogramm_poly))
                        continue;

                    // собираем wave_vector
                    Eigen::VectorXd wave_vec(T);
                    for (int t = 0; t < T; ++t)
                        wave_vec[t] = wave_data[t][i][x];

                    // собираем матрицу basis (n_basis × T)
                    Eigen::MatrixXd B(n_basis, T);
                    for (int b = 0; b < n_basis; ++b)
                        for (int t = 0; t < T; ++t)
                            B(b, t) = fk_data[b][t][i][x];

                    // вычисляем коэффициенты и погрешность
                    auto coefs = approximate_with_non_orthogonal_basis_orto(wave_vec, B);
                    Eigen::VectorXd approx = B.transpose() * coefs;
                    double err = std::sqrt((wave_vec - approx).squaredNorm() / T);

                    row.push_back({ pt, coefs, err });
                }
                return row;
            }
        ));
    }

    // собираем всё в statistics_orto
    for (auto& f : futs) {
        auto partial = f.get();
        if (!partial.empty())
            statistics_orto.push_back(std::move(partial));
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

    std::string filename_orto = "case_statistics_hd_y_" + basis + bath + "_o.json";

    save_coefficients_json(filename_orto, statistics_orto);

}