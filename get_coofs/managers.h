#ifndef MANAGERS_H
#define MANAGERS_H

#include <string>
#include <tuple>
#include <vector>
#include "stable_data_structs.h"

// ����� ��� ������ � ������� basis, ������������� � NetCDF-������
class BasisManager {
public:
    std::string folder; // ���� � �������� � basis-������� (NetCDF-�����)

    explicit BasisManager(const std::string& folder_) : folder(folder_) {}

    // ������� ������ ������ basis ��� ������� [y_start, y_end)
    // ���������� 4D ������: [num_files][T][region_height][X]
    std::vector<std::vector<std::vector<std::vector<double>>>> get_fk_region(int y_start, int y_end);

    std::size_t basis_count() const;
};

// ����� ��� ������ � ������������� (Wave data)
class WaveManager {
public:
    std::string nc_file; // ���� � NetCDF-����� � �������������

    explicit WaveManager(const std::string& nc_file_) : nc_file(nc_file_) {}

    // ������� �������� ������ ���������� "height" ��� ������� [y_start, y_end)
    // ���������� 3D ������: [T][region_height][X]
    std::vector<std::vector<std::vector<double>>> load_mariogramm_by_region(int y_start, int y_end);

    std::tuple<std::size_t, std::size_t, std::size_t> get_dimensions() const;
};

#endif // MANAGERS_H