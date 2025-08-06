#ifndef APPROX_ORTO_H
#define APPROX_ORTO_H

#include <vector>
#include <Eigen/Dense>

using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

Vector approximate_with_non_orthogonal_basis_orto(const Vector& x, const Matrix& f_k);


std::vector<double> approximate_with_non_orthogonal_basis_orto_std(
    const std::vector<double>& vector,
    const std::vector<std::vector<double>>& basis);

#endif // APPROX_ORTO_H
