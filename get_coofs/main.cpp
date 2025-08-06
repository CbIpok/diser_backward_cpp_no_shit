#include <iostream>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>
#include "approx_orto.h"
#include "stable_data_structs.h"
#include "statistics.h"


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
        std::cerr << "Ошибка копирования папки: " << e.what() << std::endl;
        return false;
    }
    return true;
}

bool deleteFolder(const std::string& folder) {
    try {
        fs::remove_all(folder);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Ошибка удаления папки: " << e.what() << std::endl;
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
        std::cerr << "Ошибка копирования файла: " << e.what() << std::endl;
        return false;
    }
    return true;
}


bool deleteFile(const std::string& file) {
    try {
        fs::remove(file);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Ошибка удаления файла: " << e.what() << std::endl;
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


    bool copiedFolder = copyFolder(sourceBasisFolder, destBasisFolder);
    if (!copiedFolder) {
        std::cerr << "Не удалось скопировать папку: " << sourceBasisFolder << std::endl;
        return -1;
    }


    std::string sourceWaveFile = root_folder + "/" + bath + "/" + wave + ".nc";
    std::string destWaveFile = destBathFolder + "/" + wave + ".nc";

    bool fileAlreadyExists = fileExists(destWaveFile);
    bool copiedFile = false;

    if (!fileAlreadyExists && fs::exists(sourceWaveFile)) {
        copiedFile = copyFile(sourceWaveFile, destWaveFile);
        if (!copiedFile) {
            std::cerr << "Не удалось скопировать файл: " << sourceWaveFile << std::endl;
        
            deleteFolder(destBasisFolder);
            return -1;
        }
    }


    save_and_plot_statistics(cache_folder, bath, wave, basis, area_config);


    if (!deleteFolder(destBasisFolder)) {
        std::cerr << "Не удалось удалить папку: " << destBasisFolder << std::endl;
    }


    if (copiedFile) {
        if (!deleteFile(destWaveFile)) {
            std::cerr << "Не удалось удалить файл: " << destWaveFile << std::endl;
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
int main() {

    run_tests();

    std::string root_folder = "T:/tsunami_res_folder";
    std::string cache_folder = "C:/dmitrienkomy/cache/";
    std::string bath = "y_200_2000";
    std::string wave = "gaus_single_2_h";
    std::string basis = "basis_48";
    std::vector<std::string> folderNames = {
        "basis_6"
    };

    // Инициализация конфигурации области (файл zones.json должен быть корректным)
    AreaConfigurationInfo area_config("T:/tsunami_res_folder/info/zones.json");

    // Вычисляем и сохраняем статистику аппроксимации
    for (auto& basis : folderNames)
    {
        runWithPrePost(root_folder, cache_folder, bath, wave, basis, area_config);
    }
    return 0;
}
#endif