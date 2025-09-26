#ifndef MANAGERS_H
#define MANAGERS_H

#include <string>
#include <vector>
#include "stable_data_structs.h"

// Класс для работы с файлами basis, расположенными в NetCDF-формате
class BasisManager {
public:
    std::string folder; // путь к каталогу с basis-файлами (NetCDF-файлы)

    explicit BasisManager(const std::string& folder_) : folder(folder_) {}

    // Загружает данные basis для диапазона [y_start, y_end)
    // Возвращает 4D массив: [num_files][T][region_height][X]
    std::vector<std::vector<std::vector<std::vector<double>>>> get_fk_region(int y_start, int y_end);

    // Количество доступных basis-файлов
    std::size_t basis_count() const;
};

// Класс для работы с волновыми данными (Wave data)
class WaveManager {
public:
    std::string nc_file; // путь к NetCDF-файлу с волновыми данными

    explicit WaveManager(const std::string& nc_file_) : nc_file(nc_file_) {}

    // Загружает значения переменной "height" для диапазона [y_start, y_end)
    // Возвращает 3D массив: [T][region_height][X]
    std::vector<std::vector<std::vector<double>>> load_mariogramm_by_region(int y_start, int y_end);
};

#endif // MANAGERS_H
