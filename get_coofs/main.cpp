#include <iostream>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>
#include "approx_orto.h"
#include "stable_data_structs.h"
#include "statistics.h"

// Для удобства
namespace fs = std::filesystem;

// Функция копирования папки (рекурсивно)
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

// Функция удаления папки
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

// Функция копирования файла
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

// Функция удаления файла
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

// Функция проверки существования файла
bool fileExists(const std::string& file) {
    return fs::exists(file);
}

// Предполагается, что AreaConfigurationInfo и функция save_and_plot_statistics уже определены
// Например:
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
        std::cerr << "Не удалось скопировать папку: " << sourceBasisFolder << std::endl;
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
            std::cerr << "Не удалось скопировать файл: " << sourceWaveFile << std::endl;
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
// Функция для проведения тестов
void run_tests() {
    // Размерности для тестовых случаев
    std::vector<int> dimensions = { 3, 4, 5, 6, 8 };
    // Порог допуска (например, 1e-6)
    double tol = 1e-6;
    bool all_passed = true;

    std::cout << "tests (approximate_with_non_orthogonal_basis_orto):\n";

    for (int n : dimensions) {
        // Генерируем случайную квадратную матрицу n x n,
        // представляющую базис (каждая строка – базисный вектор).
        Eigen::MatrixXd M;
        // Обеспечиваем обратимость (регенерируем, если определитель слишком мал)
        do {
            M = Eigen::MatrixXd::Random(n, n);
        } while (std::abs(M.determinant()) < 1e-3);

        // Генерируем случайный вектор коэффициентов c длины n.
        Eigen::VectorXd c = Eigen::VectorXd::Random(n);
        // Вычисляем вектор x как линейную комбинацию базисных векторов:
        // x = c[0]*M.row(0) + ... + c[n-1]*M.row(n-1).
        // Чтобы получить столбцовый вектор x, вычисляем:
        Eigen::VectorXd x = M.transpose() * c;

        // Вычисляем коэффициенты с использованием функции аппроксимации.
        Eigen::VectorXd b = approximate_with_non_orthogonal_basis_orto(x, M);

        // Считаем ошибку (норма разности)
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

    // Запускаем тесты, если необходимо
    run_tests();
    // Параметры проекта
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