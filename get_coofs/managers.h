#ifndef MANAGERS_H
#define MANAGERS_H

#include <string>
#include <vector>

// Функция для чтения NetCDF файла с использованием дочернего процесса.
// Данные переупорядочены в порядке: [region_height][X][T] и возвращаются в виде плоского вектора<double>.
std::vector<double> read_nc_file(const std::string& filename, int y_start, int y_end);

// Класс для загрузки basis (NetCDF файлы)
class BasisManager {
public:
    std::string folder; // Путь к директории с basis файлами

    explicit BasisManager(const std::string& folder_) : folder(folder_) {}

    // Возвращает вектор, где для каждого файла из директории возвращается
    // плоский вектор<double> с переупорядоченными данными для региона [y_start, y_end).
    std::vector<std::vector<double>> get_fk_region(int y_start, int y_end);
};

// Класс для загрузки волновых данных (Wave data)
class WaveManager {
public:
    std::string nc_file; // Путь к NetCDF файлу с волновыми данными

    explicit WaveManager(const std::string& nc_file_) : nc_file(nc_file_) {}

    // Загружает данные переменной "height" для региона [y_start, y_end)
    // и возвращает их в виде плоского вектора<double> с переупорядочиванием.
    std::vector<double> load_mariogramm_by_region(int y_start, int y_end);
};

#endif // MANAGERS_H
