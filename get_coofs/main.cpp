#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <Eigen/Dense>
#include "approx_orto.h"
#include "stable_data_structs.h"
#include "statistics.h"
#include <opencv2/highgui.hpp>    // for imshow, namedWindow, waitKey
#include <opencv2/imgcodecs.hpp> 

namespace fs = std::filesystem;


bool copyFolder(const std::string& source, const std::string& destination) {
    try {
        fs::create_directories(destination);
        for (const auto& entry : fs::recursive_directory_iterator(source)) {
            const auto& path = entry.path();
            auto relativePath = fs::relative(path, source);
            fs::copy(path, fs::path(destination) / relativePath, fs::copy_options::recursive | fs::copy_options::overwrite_existing);
        }
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Failed to copy folder: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool deleteFolder(const std::string& folder) {
    try {
        fs::remove_all(folder);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Failed to delete folder: " << e.what() << std::endl;
        return false;
    }
    return true;
}


bool copyFile(const std::string& source, const std::string& destination) {
    try {
        fs::create_directories(fs::path(destination).parent_path());
        fs::copy_file(source, destination, fs::copy_options::overwrite_existing);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Failed to copy file: " << e.what() << std::endl;
        return false;
    }
    return true;
}


bool deleteFile(const std::string& file) {
    try {
        fs::remove(file);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Failed to delete file: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool fileExists(const std::string& file) {
    return fs::exists(file);
}



int runWithPrePost(const std::string& root_folder,
    const std::string& cache_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config) {

    std::string sourceBasisFolder = root_folder + "/" + bath + "/" + basis;
    std::string destBathFolder = cache_folder + "/" + bath;
    std::string destBasisFolder = destBathFolder + "/" + basis;

    bool shouldCopy = root_folder != cache_folder;
    bool copiedFolder = false;

    if (shouldCopy) {
        copiedFolder = copyFolder(sourceBasisFolder, destBasisFolder);
        if (!copiedFolder) {
            std::cerr << "Unable to copy folder: " << sourceBasisFolder << std::endl;
            return -1;
        }
    }
    else if (!fs::exists(destBasisFolder)) {
        std::cerr << "Folder not found: " << destBasisFolder << std::endl;
        return -1;
    }

    std::string sourceWaveFile = root_folder + "/" + bath + "/" + wave + ".nc";
    std::string destWaveFile = destBathFolder + "/" + wave + ".nc";

    bool copiedFile = false;

    if (shouldCopy) {
        bool fileAlreadyExists = fileExists(destWaveFile);
        if (!fileAlreadyExists && fs::exists(sourceWaveFile)) {
            copiedFile = copyFile(sourceWaveFile, destWaveFile);
            if (!copiedFile) {
                std::cerr << "Unable to copy file: " << sourceWaveFile << std::endl;

                if (copiedFolder) {
                    deleteFolder(destBasisFolder);
                }
                return -1;
            }
        }
    }
    else if (!fileExists(destWaveFile)) {
        std::cerr << "File not found: " << destWaveFile << std::endl;
        return -1;
    }

    save_and_plot_statistics(cache_folder, bath, wave, basis, area_config);

    if (shouldCopy && copiedFolder) {
        if (!deleteFolder(destBasisFolder)) {
            std::cerr << "Failed to delete folder: " << destBasisFolder << std::endl;
        }
    }

    if (shouldCopy && copiedFile) {
        if (!deleteFile(destWaveFile)) {
            std::cerr << "Failed to delete file: " << destWaveFile << std::endl;
        }
    }

    return 0;
}

void run_tests() {

    std::vector<int> dimensions = { 3, 4, 5, 6, 8 };

    double tol = 1e-6;
    bool all_passed = true;

    std::cout << "tests (approximate_with_non_orthogonal_basis_orto):\n";

    for (int n : dimensions) {
    
        Eigen::MatrixXd M;
    
        do {
            M = Eigen::MatrixXd::Random(n, n);
        } while (std::abs(M.determinant()) < 1e-3);

   
        Eigen::VectorXd c = Eigen::VectorXd::Random(n);
  
        Eigen::VectorXd x = M.transpose() * c;

    
        Eigen::VectorXd b = approximate_with_non_orthogonal_basis_orto(x, M);

        double error = (b - c).norm();
        std::cout << "dim " << n << ": err = " << error;
        if (error < tol) {
            std::cout << " [PASSED]\n";
        }
        else {
            std::cout << " [FAILED]\n";
            all_passed = false;
        }
    }

    if (all_passed) {
        std::cout << "OK.\n";
    }
    else {
        std::cout << "FAIL.\n";
    }
}

#ifdef UNIT_TESTS
int main() {
    run_tests();
    return 0;
}
#else
int main(int argc, char* argv[]) {

    run_tests();

    if (argc < 6) {
        std::cerr << "Usage: " << argv[0]
            << " <root_folder> <cache_folder> <bath> <wave> <basis_folder1> [basis_folder2 ...]" << std::endl;
        return 1;
    }

    auto program_start = std::chrono::steady_clock::now();

    std::string root_folder = argv[1];
    std::string cache_folder = argv[2];
    std::string bath = argv[3];
    std::string wave = argv[4];
    std::vector<std::string> folderNames;
    for (int i = 5; i < argc; ++i) {
        folderNames.emplace_back(argv[i]);
    }

    // Initialize the area configuration (zones.json must be valid)
    AreaConfigurationInfo area_config("T:/tsunami_res_folder/info/zones_new.json");
    //cv::Mat img(area_config.height, area_config.width, CV_8UC3, cv::Scalar(0, 0, 0));

    //// Draw the polygon boundary
    //area_config.draw(img, /*BGR color*/ cv::Scalar(0, 255, 0), /*thickness*/ 2);

    //// Show the result
    //cv::namedWindow("Mariogramm Area", cv::WINDOW_NORMAL);
    //cv::imshow("Mariogramm Area", img);
    //cv::waitKey(0);

    for (auto& basis : folderNames)
    {
        runWithPrePost(root_folder, cache_folder, bath, wave, basis, area_config);
    }

    auto program_end = std::chrono::steady_clock::now();
    auto elapsed_seconds = std::chrono::duration<double>(program_end - program_start);
    std::cout << "Program completed in " << elapsed_seconds.count() << " seconds" << std::endl;
    return 0;
}
#endif
