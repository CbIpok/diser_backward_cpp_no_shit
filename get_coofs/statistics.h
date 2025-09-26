// statistics.h
#ifndef STATISTICS_H
#define STATISTICS_H

#include <string>
#include <vector>
#include <Eigen/Dense>
#include "stable_data_structs.h"
#include "managers.h"

// ����� ��������� ��� �������� ������������� � ������ �������������
struct CoefficientData {
    Eigen::VectorXd coefs;
    double aprox_error;
};

// ��� ��� �������� ������ �� ���� ��������: ��������� ������ �������� CoefficientData
using CoeffMatrix = std::vector<std::vector<CoefficientData>>;

// ������� ��� ���������� ���������� ������������� �� ����� �������
void calculate_statistics(const std::string& root_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config,
    CoeffMatrix& statistics_orto,
    std::vector<int>& y_indices);

// ������� ��� ���������� ���������� � JSON-����
void save_and_plot_statistics(const std::string& root_folder,
    const std::string& bath,
    const std::string& wave,
    const std::string& basis,
    const AreaConfigurationInfo& area_config);

#endif // STATISTICS_H