#ifndef MANAGERS_H
#define MANAGERS_H

#include <string>
#include <vector>

// ������� ��� ������ NetCDF ����� � �������������� ��������� ��������.
// ������ ��������������� � �������: [region_height][X][T] � ������������ � ���� �������� �������<double>.
std::vector<double> read_nc_file(const std::string& filename, int y_start, int y_end);

// ����� ��� �������� basis (NetCDF �����)
class BasisManager {
public:
    std::string folder; // ���� � ���������� � basis �������

    explicit BasisManager(const std::string& folder_) : folder(folder_) {}

    // ���������� ������, ��� ��� ������� ����� �� ���������� ������������
    // ������� ������<double> � ������������������ ������� ��� ������� [y_start, y_end).
    std::vector<std::vector<double>> get_fk_region(int y_start, int y_end);
};

// ����� ��� �������� �������� ������ (Wave data)
class WaveManager {
public:
    std::string nc_file; // ���� � NetCDF ����� � ��������� �������

    explicit WaveManager(const std::string& nc_file_) : nc_file(nc_file_) {}

    // ��������� ������ ���������� "height" ��� ������� [y_start, y_end)
    // � ���������� �� � ���� �������� �������<double> � �������������������.
    std::vector<double> load_mariogramm_by_region(int y_start, int y_end);
};

#endif // MANAGERS_H
