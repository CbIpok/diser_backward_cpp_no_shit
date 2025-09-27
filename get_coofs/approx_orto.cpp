#include "approx_orto.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>

// Функция для скалярного произведения двух векторов с использованием Eigen
inline double dot_product(const Vector& v1, const Vector& v2) {
    return v1.dot(v2);
}

Matrix gram_schmidt(const Matrix& vectors) {
    size_t n = vectors.rows();
    Matrix orthogonal_vectors = Matrix::Zero(vectors.rows(), vectors.cols());

    for (size_t i = 0; i < n; ++i) {
        Vector new_vector = vectors.row(i);
        if (i == 0) {
            orthogonal_vectors.row(0) = new_vector;
        }
        else {
            for (size_t j = 0; j < i; ++j) {

                double denominator = dot_product(orthogonal_vectors.row(j), orthogonal_vectors.row(j));

                // Проверка деления на ноль (или слишком малое значение)
                if (std::abs(denominator) < 1e-10) {
                    throw std::runtime_error("Division by zero: normalized vector is too close to zero.");
                }
                double scale = dot_product(new_vector, orthogonal_vectors.row(j)) / denominator;
                // Проверка scale на NaN
                if (std::isnan(scale)) {
                    throw std::runtime_error("NaN encountered while calculating scaling factor.");
                }
                new_vector -= scale * orthogonal_vectors.row(j);

                // Проверка каждого элемента new_vector на NaN
                for (int k = 0; k < new_vector.size(); ++k) {
                    if (std::isnan(new_vector[k])) {
                        throw std::runtime_error("NaN encountered in vector after subtraction.");
                    }
                }
            }
            orthogonal_vectors.row(i) = new_vector;
        }
        // Альтернативно, можно проверить всю строку с использованием встроенной функции Eigen:
        if (orthogonal_vectors.row(i).hasNaN()) {
            throw std::runtime_error("NaN detected in orthogonal row.");
        }
    }
    return orthogonal_vectors;
}


// Функция для разложения вектора по ортогональному базису с использованием Eigen
Vector decompose_vector(const Vector& v, const Matrix& orthogonal_basis) {
    Vector coefficients(orthogonal_basis.rows());
    for (size_t i = 0; i < orthogonal_basis.rows(); ++i) {
        double denominator = dot_product(orthogonal_basis.row(i), orthogonal_basis.row(i));
        coefficients[i] = (denominator != 0) ? dot_product(v, orthogonal_basis.row(i)) / denominator : 0;
    }
    return coefficients;
}

// Вычисление l_k_i для конкретных индексов k и i
inline double compute_l_k_i(const Vector& f_k_i, const Vector& e_i) {
    double dot_fk_ei = dot_product(f_k_i, e_i);
    double dot_ei_ei = dot_product(e_i, e_i);
    return (dot_ei_ei == 0) ? 0 : -dot_fk_ei / dot_ei_ei;
}

// Итеративная версия вычисления матрицы F
Matrix compute_F_matrix(const Matrix& l) {
    size_t n = l.rows();
    Matrix F_matrix = Matrix::Zero(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            double sum_part = 0;
            for (size_t k = i + 1; k < j; ++k) {
                sum_part += l(j, k) * F_matrix(k, i);
            }
            F_matrix(j, i) = l(j, i) + sum_part;
        }
    }
    return F_matrix;
}

// Вычисление вектора b
Vector compute_bi(size_t k, const Vector& a_k, const Matrix& l) {
    Vector b = Vector::Zero(a_k.size());
    Matrix F_matrix = compute_F_matrix(l);
    b[k] = a_k[k];
    b[k - 1] = a_k[k - 1] + a_k[k] * F_matrix(k, k - 1);
    b[k - 2] = a_k[k - 2] + a_k[k - 1] * F_matrix(k - 1, k - 2) + a_k[k] * F_matrix(k, k - 2);
    for (int i = static_cast<int>(k) - 3; i >= 0; --i) {
        double sum_part = 0;
        for (size_t j = i + 1; j <= k; ++j) {
            sum_part += a_k[j] * F_matrix(j, i);
        }
        b[i] = a_k[i] + sum_part;
    }
    return b;
}

// Основная функция аппроксимации (ортогонализованный метод)
Vector approximate_with_non_orthogonal_basis_orto(const Vector& x, const Matrix& f_k) {

    Matrix e_i = gram_schmidt(f_k);

    // Разложение вектора x по ортогональному базису
    Vector a_k = decompose_vector(x, e_i);

    // Вычисление матрицы l_k_i
    Matrix l_k_i(f_k.rows(), f_k.cols());
    for (size_t k = 0; k < f_k.rows(); ++k) {
        for (size_t i = 0; i < e_i.rows(); ++i) {
            l_k_i(k, i) = compute_l_k_i(f_k.row(k), e_i.row(i));
        }
    }
    // Вычисляем результирующий вектор коэффициентов
    size_t k = a_k.size() - 1;
    Vector b = compute_bi(k, a_k, l_k_i);
    return b;
}

// Обёртка для работы со стандартными векторами
std::vector<double> approximate_with_non_orthogonal_basis_orto_std(
    const std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis) {
    // Преобразуем std::vector в Eigen::VectorXd
    Vector x = Eigen::Map<const Vector>(vector.data(), vector.size());
    // Преобразуем двумерный std::vector в Eigen::MatrixXd
    size_t rows = basis.size();
    size_t cols = basis[0].size();
    Matrix f_k(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            f_k(i, j) = basis[i][j];
    // Вычисляем коэффициенты
    Vector result = approximate_with_non_orthogonal_basis_orto(x, f_k);
    return std::vector<double>(result.data(), result.data() + result.size());
}
