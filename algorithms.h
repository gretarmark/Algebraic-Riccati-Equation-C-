#ifndef ALGORITHMS_H
#define ALGORITHMS_H

#include <Eigen/Dense>

using namespace Eigen;

// Function to solve ARE using Schur Decomposition Method
MatrixXd solveARE_Schur(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R);

// Function to solve ARE using Eigenvalue Decomposition
Eigen::MatrixXd solveARE_Eigenvalue(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);

// Function to solve ARE using Newton's Method
Eigen::MatrixXd solveARE_Newton(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, int maxIterations, double tolerance);

// Function declaration for solving a simple sum
int sum(int a, int b);

#endif
