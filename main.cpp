#include <iostream>
#include "algorithms.h"

using namespace std;
using namespace Eigen;

int main() {
    int a = 5;
    int b = 10;

    // Call the sum function from algorithms.cpp
    int result = sum(a, b);
    //int result = 5;

    // Print the result
    cout << "The sum of " << a << " and " << b << " is: " << result << endl;

    // Define the matrices A, B, Q, R
    Eigen::MatrixXd A(2, 2);
    Eigen::MatrixXd B(2, 1);
    Eigen::MatrixXd Q(2, 2);
    Eigen::MatrixXd R(1, 1);


    A << 1, 2, 3, 4;
    B << 1, 0;
    Q << 2, 1, 1, 2;
    R << 1;

    cout << "Solving ARE using Schur Decomposition Method" << endl;
    MatrixXd P_schur = solveARE_Schur(A, B, Q, R);
    cout << "P (Eigenvalue Method):" << endl;
    cout << P_schur << endl;

    cout << "\nSolving ARE using Eigenvalue Decomposition..." << endl;
    MatrixXd P_eigenvalue = solveARE_Eigenvalue(A, B, Q, R);
    cout << "P (Eigenvalue Method):" << endl;
    cout << P_eigenvalue << endl;

    cout << "\nSolving ARE using Newton's Method..." << endl;
    MatrixXd P_newton = solveARE_Newton(A, B, Q, R, 100, 1e-6);
    cout << "P (Newton's Method):" << endl;
    cout << P_newton << endl;

    return 0;
}
