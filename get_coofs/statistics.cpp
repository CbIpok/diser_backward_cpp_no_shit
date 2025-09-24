// statistics.cpp
#include "statistics.h"
#include "approx_orto.h"
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <sstream>
#include <future>
#include <algorithm>
#include <limits>
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

    std::size_t wave_T = 0, wave_Y = 0, wave_X = 0;
    if (!wave_manager.get_dimensions(wave_T, wave_Y, wave_X)) {
        std::cerr << "failed to query wave dimensions for " << wave_manager.nc_file << "\n";
        return;
    }

    std::size_t basis_count = basis_manager.basis_count();
    if (basis_count == 0) {
        std::cerr << "no basis files found in " << basis_manager.folder << "\n";
        return;
    }

    int T = static_cast<int>(wave_T);
    int W = static_cast<int>(wave_X);
    int data_height = static_cast<int>(wave_Y);
    int n_basis = static_cast<int>(basis_count);

    if (T <= 0 || W <= 0 || data_height <= 0) {
        std::cerr << "invalid data dimensions\n";
        return;
    }

    minX = std::max(0, minX);
    maxX = std::min(maxX, W - 1);
    minY = std::max(0, minY);
    maxY = std::min(maxY, data_height);
    if (minX > maxX || minY >= maxY) {
        std::cerr << "polygon bounds do not intersect data domain\n";
        return;
    }

    int H = maxY - minY;
    if (H <= 0) {
        return;
    }

    constexpr std::size_t MAX_MEMORY_BYTES = 45ULL * 1024ULL * 1024ULL * 1024ULL; // 50 GB
    std::size_t bytes_per_row = static_cast<std::size_t>(n_basis + 1)
        * static_cast<std::size_t>(T) * static_cast<std::size_t>(W) * sizeof(double);
    std::size_t rows_per_band = bytes_per_row > 0 ? MAX_MEMORY_BYTES / bytes_per_row : 0;
    if (rows_per_band == 0) {
        rows_per_band = 1;
    }
    if (rows_per_band > static_cast<std::size_t>(H)) {
        rows_per_band = static_cast<std::size_t>(H);
    }
    int band_height = static_cast<int>(std::min<std::size_t>(rows_per_band,
        static_cast<std::size_t>(std::numeric_limits<int>::max())));
    band_height = std::max(1, band_height);

    struct ProcessPoint {
        int local_row;
        int x;
    };

    for (int band_start = 0; band_start < H; band_start += band_height) {
        int band_end = std::min(H, band_start + band_height);

        auto fk_data = basis_manager.get_fk_region(minY + band_start, minY + band_end);
        auto wave_data = wave_manager.load_mariogramm_by_region(minY + band_start, minY + band_end);
        if (fk_data.empty() || wave_data.empty()) {
            continue;
        }

        int bandH = band_end - band_start;
        if (bandH <= 0) {
            continue;
        }

        std::vector<ProcessPoint> band_points;
        std::size_t width_range = static_cast<std::size_t>(std::max(0, maxX - minX + 1));
        band_points.reserve(static_cast<std::size_t>(bandH) * width_range);
        std::vector<std::size_t> row_point_counts(static_cast<std::size_t>(bandH), 0);

        for (int local_row = 0; local_row < bandH; ++local_row) {
            int global_y = minY + band_start + local_row;
            for (int x = minX; x <= maxX; ++x) {
                if (x < 0 || x >= W) {
                    continue;
                }
                Point2i pt(x, global_y);
                if (!bg::within(pt, area_config.mariogramm_poly)) {
                    continue;
                }
                band_points.push_back({ local_row, x });
                ++row_point_counts[static_cast<std::size_t>(local_row)];
            }
        }

        if (band_points.empty()) {
            continue;
        }

        constexpr int THREADS = 48;
        int thread_count = std::min<int>(THREADS, static_cast<int>(band_points.size()));
        if (thread_count <= 0) {
            continue;
        }

        std::size_t chunk_size = (band_points.size() + static_cast<std::size_t>(thread_count) - 1)
            / static_cast<std::size_t>(thread_count);

        CoeffMatrix band_results(bandH);
        for (int row = 0; row < bandH; ++row) {
            band_results[row].reserve(row_point_counts[static_cast<std::size_t>(row)]);
        }

        std::vector<CoeffMatrix> thread_local_results(static_cast<std::size_t>(thread_count), CoeffMatrix(bandH));
        std::vector<std::future<void>> futs;
        futs.reserve(thread_count);

        for (int thread_idx = 0; thread_idx < thread_count; ++thread_idx) {
            std::size_t start_idx = static_cast<std::size_t>(thread_idx) * chunk_size;
            if (start_idx >= band_points.size()) {
                break;
            }
            std::size_t end_idx = std::min(band_points.size(), start_idx + chunk_size);

            futs.emplace_back(std::async(std::launch::async,
                [&, thread_idx, start_idx, end_idx]() {
                    auto& local = thread_local_results[static_cast<std::size_t>(thread_idx)];
                    std::vector<std::size_t> local_counts(static_cast<std::size_t>(bandH), 0);

                    for (std::size_t idx = start_idx; idx < end_idx; ++idx) {
                        ++local_counts[static_cast<std::size_t>(band_points[idx].local_row)];
                    }

                    for (int row = 0; row < bandH; ++row) {
                        std::size_t count = local_counts[static_cast<std::size_t>(row)];
                        if (count > 0) {
                            local[row].reserve(count);
                        }
                    }

                    Eigen::VectorXd wave_vec(T);
                    Eigen::MatrixXd B(n_basis, T);

                    for (std::size_t idx = start_idx; idx < end_idx; ++idx) {
                        int local_row = band_points[idx].local_row;
                        int x = band_points[idx].x;
                        int global_y = minY + band_start + local_row;

                        for (int t = 0; t < T; ++t) {
                            wave_vec[t] = wave_data[t][local_row][x];
                        }
                        for (int b = 0; b < n_basis; ++b) {
                            for (int t = 0; t < T; ++t) {
                                B(b, t) = fk_data[b][t][local_row][x];
                            }
                        }

                        auto coefs = approximate_with_non_orthogonal_basis_orto(wave_vec, B);
                        Eigen::VectorXd approx = B.transpose() * coefs;
                        double err = std::sqrt((wave_vec - approx).squaredNorm() / T);

                        local[local_row].push_back({ Point2i(x, global_y), coefs, err });
                    }
                }));
        }

        for (auto& f : futs) {
            f.get();
        }

        for (int row = 0; row < bandH; ++row) {
            auto& dst_row = band_results[row];
            for (auto& local_matrix : thread_local_results) {
                auto& src_row = local_matrix[row];
                if (!src_row.empty()) {
                    std::move(src_row.begin(), src_row.end(), std::back_inserter(dst_row));
                    src_row.clear();
                }
            }
        }

        for (auto& row : band_results) {
            if (!row.empty()) {
                statistics_orto.push_back(std::move(row));
            }
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