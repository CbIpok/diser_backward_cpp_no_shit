#include <iostream>
#include <vector>
#include <filesystem>
#include <chrono>
#include <Eigen/Dense>
#include "approx_orto.h"
#include "stable_data_structs.h"
#include "statistics.h"

// For convenience
namespace fs = std::filesystem;

// Function to copy a folder (recursively)
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
        std::cerr << "Folder copy error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// Function to delete a folder
bool deleteFolder(const std::string& folder) {
    try {
        fs::remove_all(folder);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Folder deletion error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// Function to copy a file
bool copyFile(const std::string& source, const std::string& destination) {
    try {
        fs::create_directories(fs::path(destination).parent_path());
        fs::copy_file(source, destination, fs::copy_options::overwrite_existing);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "File copy error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// Function to delete a file
bool deleteFile(const std::string& file) {
    try {
        fs::remove(file);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "File deletion error: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// Function to check if a file or folder exists
bool fileExists(const std::string& file) {
    return fs::exists(file);
}

// It is assumed that AreaConfigurationInfo and the function save_and_plot_statistics are already defined
// For example:
// class AreaConfigurationInfo { /* ... */ };
// void save_and_plot_statistics(const std::string&, const std::string&, const std::string&, const std::string&, const AreaConfigurationInfo&);

int runWithPrePost(const std::string& root_folder,
    const std::string& cache_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config) {
    // Формируем пути для копирования папки basis
    std::string sourceBasisFolder = root_folder + "/" + bath + "/" + basis;
    std::string destBathFolder = cache_folder + "/" + bath;
    std::string destBasisFolder = destBathFolder + "/" + basis;

    // Копируем папку basis
    bool copiedFolder = copyFolder(sourceBasisFolder, destBasisFolder);
    if (!copiedFolder) {
        std::cerr << "error copy folder: " << sourceBasisFolder << std::endl;
        return -1;
    }

    // Формируем пути для файла wave
    std::string sourceWaveFile = root_folder + "/" + bath + "/" + wave + ".nc";
    std::string destWaveFile = destBathFolder + "/" + wave + ".nc";

    bool fileAlreadyExists = fileExists(destWaveFile);
    bool copiedFile = false;

    // Если файл не существует в целевой папке и исходный файл есть – копируем
    if (!fileAlreadyExists && fs::exists(sourceWaveFile)) {
        copiedFile = copyFile(sourceWaveFile, destWaveFile);
        if (!copiedFile) {
            std::cerr << "error copy file: " << sourceWaveFile << std::endl;
            // Если не удалось скопировать файл, удаляем ранее скопированную папку
            deleteFolder(destBasisFolder);
            return -1;
        }
    }

    // Выполняем основную функцию
    save_and_plot_statistics(cache_folder, bath, wave, basis, area_config);

    // Удаляем скопированную папку basis из кэша
    if (!deleteFolder(destBasisFolder)) {
        std::cerr << "Не удалось удалить папку: " << destBasisFolder << std::endl;
    }

    // Если файл был скопирован (то есть его не было заранее) – удаляем его
    if (copiedFile) {
        if (!deleteFile(destWaveFile)) {
            std::cerr << "Не удалось удалить файл: " << destWaveFile << std::endl;
        }
    }

    return 0;
}

void run_tests() {
    // Размерности для тестовых случаев
    std::vector<int> dimensions = { 3, 4, 5, 6, 8 };
    // Допустимая погрешность
    double tol = 1e-6;
    bool all_passed = true;

    std::cout << "tests (approximate_with_non_orthogonal_basis_orto) с постоянным базисом размера n x 2n:\n";

    for (int n : dimensions) {
        // Построение константного линейно независимого базиса M размера n x (2*n).
        // Каждая строка i имеет единицу в столбце i и в столбце (n + i)
        Eigen::MatrixXd M = Eigen::MatrixXd::Zero(n, 2 * n);
        for (int i = 0; i < n; ++i) {
            M(i, i) = 1.0;
            M(i, n + i) = 1.0;
        }

        // Генерация случайного коэффициентного вектора c длины n.
        Eigen::VectorXd c = Eigen::VectorXd::Random(n);
        // Вычисление вектора x как линейной комбинации строк базиса:
        // x = c[0]*M.row(0) + ... + c[n-1]*M.row(n-1)
        // При этом x получается как M.transpose() * c, и его размерность будет 2*n.
        Eigen::VectorXd x = M.transpose() * c;

        // Вычисление коэффициентов с помощью функции аппроксимации.
        Eigen::VectorXd b = approximate_with_non_orthogonal_basis_orto(x, M);

        // Вычисление ошибки (норма разности между найденными и исходными коэффициентами)
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
    // Check for required arguments: bath, wave, basis
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " bath wave basis" << std::endl;
        return 1;
    }

    // Read command line arguments
    std::string bath = argv[1];
    std::string wave = argv[2];
    std::string basis = argv[3];

    // Other parameters can be fixed or also obtained from arguments
    std::string root_folder = "T:/tsunami_res_folder";
    std::string cache_folder = "C:/dmitrienkomy/cache";

    // Initialize area configuration (zones.json must be correct)
    AreaConfigurationInfo area_config("T:/tsunami_res_folder/info/zones.json");

    // If needed, run run_tests(), for example for debugging:
    run_tests();

    // Execute processing for given parameters
    // You can use either runWithPrePost or directly save_and_plot_statistics
    // Example usage of runWithPrePost:
    if (runWithPrePost(root_folder, cache_folder, bath, wave, basis, area_config) != 0) {
        std::cerr << "run error in runWithPrePost.\n";
        return 1;
    }
    //save_and_plot_statistics(cache_folder, "x_200_2000", "gaus_double_1_2", "basis_6", area_config);
    return 0;
}
#endif
