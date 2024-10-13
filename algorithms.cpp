#include <iostream>
#include "algorithms.h"


MatrixXd solveARE_Schur(const MatrixXd &A, const MatrixXd &B, const MatrixXd &Q, const MatrixXd &R) {
    int n = A.rows();  // Get size of A

    // Check if R is invertible
    if (R.determinant() == 0) {
        std::cerr << "Matrix R is singular; cannot invert." << std::endl;
        return MatrixXd::Zero(n, n);
    }

    // Step 1: Construct the Hamiltonian matrix
    MatrixXd R_inv = R.inverse();
    MatrixXd BRB = B * R_inv * B.transpose();

    // Create the Hamiltonian matrix
    MatrixXd H(2 * n, 2 * n);
    H.topLeftCorner(n, n) = A;
    H.topRightCorner(n, n) = -BRB;
    H.bottomLeftCorner(n, n) = -Q;
    H.bottomRightCorner(n, n) = -A.transpose();

    // Print the Hamiltonian matrix
    std::cout << "Hamiltonian Matrix H:\n" << H << std::endl;

    // Step 2: Compute Schur decomposition
    RealSchur<MatrixXd> schur(H);
    MatrixXd U = schur.matrixU();
    MatrixXd T = schur.matrixT();

    // Retrieve eigenvalues from the diagonal of T
    Eigen::VectorXd eigenvalues = T.diagonal();

    // Print eigenvalues from the Schur decomposition
    std::cout << "Eigenvalues from Schur Decomposition:\n" << eigenvalues.transpose() << std::endl;

    // Step 3: Extract stable subspace (eigenvalues with negative real parts)
    MatrixXd U_11 = U.topLeftCorner(n, n);
    MatrixXd U_21 = U.bottomLeftCorner(n, n);

    // Check if U_11 is invertible
    if (U_11.determinant() == 0) {
        std::cerr << "Matrix U_11 is singular; cannot compute inverse." << std::endl;
        return MatrixXd::Zero(n, n);
    }

    // Step 4: Solve for P using P = U_21 * inv(U_11)
    MatrixXd P = U_21 * U_11.inverse();
    
    // Print the resulting matrix P
    std::cout << "P (Schur Method):\n" << P << std::endl;

    return P;
}

// Function to solve ARE using Eigenvalue Decomposition
Eigen::MatrixXd solveARE_Eigenvalue(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R) {
    // Compute the Hamiltonian matrix
    Eigen::MatrixXd Z = R.inverse() * B.transpose();
    Eigen::MatrixXd H(2 * A.rows(), 2 * A.cols());

    H << A, -B * Z,
         -Q, -A.transpose();

    // Compute the Eigenvalues and Eigenvectors
    Eigen::EigenSolver<Eigen::MatrixXd> es(H);
    Eigen::MatrixXd eigenvalues = es.eigenvalues().real();
    Eigen::MatrixXd eigenvectors = es.eigenvectors().real();

    // Separate stable and unstable eigenvalues and eigenvectors
    Eigen::MatrixXd X1(A.rows(), A.cols());
    Eigen::MatrixXd X2(A.rows(), A.cols());
    int count = 0;
    
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) < 0) {  // Select eigenvalues with negative real parts
            X1.col(count) = eigenvectors.col(i).topRows(A.rows());
            X2.col(count) = eigenvectors.col(i).bottomRows(A.rows());
            count++;
        }
    }

    if (X1.cols() != A.rows()) {
        std::cerr << "Error: Could not find enough stable eigenvectors." << std::endl;
        return Eigen::MatrixXd::Zero(A.rows(), A.cols());
    }

    // Solve P = X2 * X1^(-1)
    Eigen::MatrixXd P = X2 * X1.inverse();

    return P;
}

// Function to solve ARE using Newton's Method
Eigen::MatrixXd solveARE_Newton(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R, int maxIterations, double tolerance) {
    int n = A.rows();
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n, n); // Initial guess for P
    Eigen::MatrixXd P_next;

    if (R.determinant() == 0) {
        std::cerr << "Matrix R is singular. Cannot compute its inverse." << std::endl;
        return Eigen::MatrixXd::Constant(n, n, std::nan(""));
    }

    for (int iter = 0; iter < maxIterations; ++iter) {
        Eigen::MatrixXd R_inv = R.inverse();
        Eigen::MatrixXd K = R_inv * B.transpose() * P;
        Eigen::MatrixXd H = A - B * K;

        if (H.determinant() == 0) {
            std::cerr << "Matrix H is singular at iteration " << iter << ". Stopping..." << std::endl;
            return P;
        }

        P_next = Q + H.transpose() * P * H - P * B * R_inv * B.transpose() * P;

        if (P_next.hasNaN()) {
            std::cerr << "NaN detected in P_next at iteration " << iter << ". Stopping..." << std::endl;
            return Eigen::MatrixXd::Constant(n, n, std::nan(""));
        }

        if ((P_next - P).norm() < tolerance) {
            std::cout << "Converged in " << iter << " iterations." << std::endl;
            return P_next;
        }

        P = P_next;
    }

    std::cerr << "Max iterations reached without convergence." << std::endl;
    return Eigen::MatrixXd::Constant(n, n, std::nan(""));
}



// Function implementation for sum
int sum(int a, int b) {
    return a + b;
}
