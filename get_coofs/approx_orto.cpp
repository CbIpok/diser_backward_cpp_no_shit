#include "approx_orto.h"
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>

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

                // Ïðîâåðêà äåëåíèÿ íà íîëü (èëè ñëèøêîì ìàëîå çíà÷åíèå)
                if (std::abs(denominator) < 1e-10) {
                    std::cout << "f: " << orthogonal_vectors.row(j) << std::endl;
                    throw std::runtime_error("Äåëåíèå íà íîëü: íîðìèðîâàííûé âåêòîð ñëèøêîì áëèçîê ê íóëþ.");
                }
                double scale = dot_product(new_vector, orthogonal_vectors.row(j)) / denominator;
                // Ïðîâåðêà scale íà NaN
                if (std::isnan(scale)) {
                    throw std::runtime_error("Îáíàðóæåíî NaN ïðè âû÷èñëåíèè êîýôôèöèåíòà ìàñøòàáèðîâàíèÿ.");
                }
                new_vector -= scale * orthogonal_vectors.row(j);

                // Ïðîâåðêà êàæäîãî ýëåìåíòà new_vector íà NaN
                for (int k = 0; k < new_vector.size(); ++k) {
                    if (std::isnan(new_vector[k])) {
                        throw std::runtime_error("Îáíàðóæåíî NaN â âåêòîðå ïîñëå âû÷èòàíèÿ.");
                    }
                }
            }
            orthogonal_vectors.row(i) = new_vector;
        }
        // Àëüòåðíàòèâíî, ìîæíî ïðîâåðèòü âñþ ñòðîêó ñ èñïîëüçîâàíèåì âñòðîåííîé ôóíêöèè Eigen:
        if (orthogonal_vectors.row(i).hasNaN()) {
            throw std::runtime_error("Îáíàðóæåíî NaN â âû÷èñëåííîé îðòîãîíàëüíîé ñòðîêå.");
        }
    }
    return orthogonal_vectors;
}


// Ôóíêöèÿ äëÿ ðàçëîæåíèÿ âåêòîðà ïî îðòîãîíàëüíîìó áàçèñó ñ èñïîëüçîâàíèåì Eigen
Vector decompose_vector(const Vector& v, const Matrix& orthogonal_basis) {
    Vector coefficients(orthogonal_basis.rows());
    for (size_t i = 0; i < orthogonal_basis.rows(); ++i) {
        double denominator = dot_product(orthogonal_basis.row(i), orthogonal_basis.row(i));
        coefficients[i] = (denominator != 0) ? dot_product(v, orthogonal_basis.row(i)) / denominator : 0;
    }
    return coefficients;
}

// Âû÷èñëåíèå l_k_i äëÿ êîíêðåòíûõ èíäåêñîâ k è i
inline double compute_l_k_i(const Vector& f_k_i, const Vector& e_i) {
    double dot_fk_ei = dot_product(f_k_i, e_i);
    double dot_ei_ei = dot_product(e_i, e_i);
    return (dot_ei_ei == 0) ? 0 : -dot_fk_ei / dot_ei_ei;
}

// Èòåðàòèâíàÿ âåðñèÿ âû÷èñëåíèÿ ìàòðèöû F
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

// Âû÷èñëåíèå âåêòîðà b
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

// Îñíîâíàÿ ôóíêöèÿ àïïðîêñèìàöèè (îðòîãîíàëèçîâàííûé ìåòîä)
Vector approximate_with_non_orthogonal_basis_orto(const Vector& x, const Matrix& f_k) {

    Matrix e_i = gram_schmidt(f_k);

    // Ðàçëîæåíèå âåêòîðà x ïî îðòîãîíàëüíîìó áàçèñó
    Vector a_k = decompose_vector(x, e_i);

    // Âû÷èñëåíèå ìàòðèöû l_k_i
    Matrix l_k_i(f_k.rows(), f_k.cols());
    for (size_t k = 0; k < f_k.rows(); ++k) {
        for (size_t i = 0; i < e_i.rows(); ++i) {
            l_k_i(k, i) = compute_l_k_i(f_k.row(k), e_i.row(i));
        }
    }
    // Âû÷èñëÿåì ðåçóëüòèðóþùèé âåêòîð êîýôôèöèåíòîâ
    size_t k = a_k.size() - 1;
    Vector b = compute_bi(k, a_k, l_k_i);
    return b;
}

// Îá¸ðòêà äëÿ ðàáîòû ñî ñòàíäàðòíûìè âåêòîðàìè
std::vector<double> approximate_with_non_orthogonal_basis_orto_std(
    const std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis) {
    // Ïðåîáðàçóåì std::vector â Eigen::VectorXd
    Vector x = Eigen::Map<const Vector>(vector.data(), vector.size());
    // Ïðåîáðàçóåì äâóìåðíûé std::vector â Eigen::MatrixXd
    size_t rows = basis.size();
    size_t cols = basis[0].size();
    Matrix f_k(rows, cols);
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
            f_k(i, j) = basis[i][j];
    // Âû÷èñëÿåì êîýôôèöèåíòû
    Vector result = approximate_with_non_orthogonal_basis_orto(x, f_k);
    return std::vector<double>(result.data(), result.data() + result.size());
}
