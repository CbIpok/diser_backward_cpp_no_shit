#include <iostream>
#include <vector>
#include <filesystem>
#include <Eigen/Dense>
#include "approx_orto.h"
#include "stable_data_structs.h"
#include "statistics.h"

// Äëÿ óäîáñòâà
namespace fs = std::filesystem;

// Ôóíêöèÿ êîïèðîâàíèÿ ïàïêè (ðåêóðñèâíî)
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
        std::cerr << "Îøèáêà êîïèðîâàíèÿ ïàïêè: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// Ôóíêöèÿ óäàëåíèÿ ïàïêè
bool deleteFolder(const std::string& folder) {
    try {
        fs::remove_all(folder);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Îøèáêà óäàëåíèÿ ïàïêè: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// Ôóíêöèÿ êîïèðîâàíèÿ ôàéëà
bool copyFile(const std::string& source, const std::string& destination) {
    try {
        fs::create_directories(fs::path(destination).parent_path());
        fs::copy_file(source, destination, fs::copy_options::overwrite_existing);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Îøèáêà êîïèðîâàíèÿ ôàéëà: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// Ôóíêöèÿ óäàëåíèÿ ôàéëà
bool deleteFile(const std::string& file) {
    try {
        fs::remove(file);
    }
    catch (fs::filesystem_error& e) {
        std::cerr << "Îøèáêà óäàëåíèÿ ôàéëà: " << e.what() << std::endl;
        return false;
    }
    return true;
}

// Ôóíêöèÿ ïðîâåðêè ñóùåñòâîâàíèÿ ôàéëà
bool fileExists(const std::string& file) {
    return fs::exists(file);
}

// Ïðåäïîëàãàåòñÿ, ÷òî AreaConfigurationInfo è ôóíêöèÿ save_and_plot_statistics óæå îïðåäåëåíû
// Íàïðèìåð:
// class AreaConfigurationInfo { /* ... */ };
// void save_and_plot_statistics(const std::string&, const std::string&, const std::string&, const std::string&, const AreaConfigurationInfo&);

int runWithPrePost(const std::string& root_folder,
    const std::string& cache_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config) {
    // Ôîðìèðóåì ïóòè äëÿ êîïèðîâàíèÿ ïàïêè basis
    std::string sourceBasisFolder = root_folder + "/" + bath + "/" + basis;
    std::string destBathFolder = cache_folder + "/" + bath;
    std::string destBasisFolder = destBathFolder + "/" + basis;

    // Êîïèðóåì ïàïêó basis
    bool copiedFolder = copyFolder(sourceBasisFolder, destBasisFolder);
    if (!copiedFolder) {
        std::cerr << "Íå óäàëîñü ñêîïèðîâàòü ïàïêó: " << sourceBasisFolder << std::endl;
        return -1;
    }

    // Ôîðìèðóåì ïóòè äëÿ ôàéëà wave
    std::string sourceWaveFile = root_folder + "/" + bath + "/" + wave + ".nc";
    std::string destWaveFile = destBathFolder + "/" + wave + ".nc";

    bool fileAlreadyExists = fileExists(destWaveFile);
    bool copiedFile = false;

    // Åñëè ôàéë íå ñóùåñòâóåò â öåëåâîé ïàïêå è èñõîäíûé ôàéë åñòü – êîïèðóåì
    if (!fileAlreadyExists && fs::exists(sourceWaveFile)) {
        copiedFile = copyFile(sourceWaveFile, destWaveFile);
        if (!copiedFile) {
            std::cerr << "Íå óäàëîñü ñêîïèðîâàòü ôàéë: " << sourceWaveFile << std::endl;
            // Åñëè íå óäàëîñü ñêîïèðîâàòü ôàéë, óäàëÿåì ðàíåå ñêîïèðîâàííóþ ïàïêó
            deleteFolder(destBasisFolder);
            return -1;
        }
    }

    // Âûïîëíÿåì îñíîâíóþ ôóíêöèþ
    save_and_plot_statistics(cache_folder, bath, wave, basis, area_config);

    // Óäàëÿåì ñêîïèðîâàííóþ ïàïêó basis èç êýøà
    if (!deleteFolder(destBasisFolder)) {
        std::cerr << "Íå óäàëîñü óäàëèòü ïàïêó: " << destBasisFolder << std::endl;
    }

    // Åñëè ôàéë áûë ñêîïèðîâàí (òî åñòü åãî íå áûëî çàðàíåå) – óäàëÿåì åãî
    if (copiedFile) {
        if (!deleteFile(destWaveFile)) {
            std::cerr << "Íå óäàëîñü óäàëèòü ôàéë: " << destWaveFile << std::endl;
        }
    }

    return 0;
}
// Ôóíêöèÿ äëÿ ïðîâåäåíèÿ òåñòîâ
void run_tests() {
    // Ðàçìåðíîñòè äëÿ òåñòîâûõ ñëó÷àåâ
    std::vector<int> dimensions = { 3, 4, 5, 6, 8 };
    // Ïîðîã äîïóñêà (íàïðèìåð, 1e-6)
    double tol = 1e-6;
    bool all_passed = true;

    std::cout << "tests (approximate_with_non_orthogonal_basis_orto):\n";

    for (int n : dimensions) {
        // Ãåíåðèðóåì ñëó÷àéíóþ êâàäðàòíóþ ìàòðèöó n x n,
        // ïðåäñòàâëÿþùóþ áàçèñ (êàæäàÿ ñòðîêà – áàçèñíûé âåêòîð).
        Eigen::MatrixXd M;
        // Îáåñïå÷èâàåì îáðàòèìîñòü (ðåãåíåðèðóåì, åñëè îïðåäåëèòåëü ñëèøêîì ìàë)
        do {
            M = Eigen::MatrixXd::Random(n, n);
        } while (std::abs(M.determinant()) < 1e-3);

        // Ãåíåðèðóåì ñëó÷àéíûé âåêòîð êîýôôèöèåíòîâ c äëèíû n.
        Eigen::VectorXd c = Eigen::VectorXd::Random(n);
        // Âû÷èñëÿåì âåêòîð x êàê ëèíåéíóþ êîìáèíàöèþ áàçèñíûõ âåêòîðîâ:
        // x = c[0]*M.row(0) + ... + c[n-1]*M.row(n-1).
        // ×òîáû ïîëó÷èòü ñòîëáöîâûé âåêòîð x, âû÷èñëÿåì:
        Eigen::VectorXd x = M.transpose() * c;

        // Âû÷èñëÿåì êîýôôèöèåíòû ñ èñïîëüçîâàíèåì ôóíêöèè àïïðîêñèìàöèè.
        Eigen::VectorXd b = approximate_with_non_orthogonal_basis_orto(x, M);

        // Ñ÷èòàåì îøèáêó (íîðìà ðàçíîñòè)
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

    AreaConfigurationInfo area_config("T:/tsunami_res_folder/info/new_zones.json");
    run_tests();
    // Ïàðàìåòðû ïðîåêòà
    std::string root_folder = "T:/tsunami_res_folder";
    std::string cache_folder = "C:/dmitrienkomy/cache/";
    std::string bath = "y_200_2000";
    std::string wave = "gaus_single_2_h";
    std::string basis = "basis_48";
    std::vector<std::string> folderNames = {
        "basis_6"
    };

    // Èíèöèàëèçàöèÿ êîíôèãóðàöèè îáëàñòè (ôàéë zones.json äîëæåí áûòü êîððåêòíûì)
    AreaConfigurationInfo area_config("T:/tsunami_res_folder/info/zones.json");

    // Âû÷èñëÿåì è ñîõðàíÿåì ñòàòèñòèêó àïïðîêñèìàöèè
    for (auto& basis : folderNames)
    {
        runWithPrePost(root_folder, cache_folder, bath, wave, basis, area_config);
    }
    return 0;
}
#endif