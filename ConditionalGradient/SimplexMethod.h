#pragma once

#include "Matrix.h"



void print(Matrix<double> matrix);
void print(const std::vector<double>& matrix, int n, int m);
void print(std::vector<int> vec);



class SimplexMethod {
private:
    Matrix<double> A;
    Matrix<double> b;
    Matrix<double> c;
    Matrix<double> B_Nk;
    Matrix<double> d_Nk;
    std::vector<int> Nk_ind;
    std::vector<int> Lk_ind;
    std::vector<int> M_ind;
    std::vector<int> N_ind;
    std::vector<bool> verify;
    int index_jk;
    int index_i;
    Matrix<double> referenceVector;
    void add();
    bool calcD_1();
    bool calcD_2();
    void set_new_B();
    double theta(Matrix<double> a, Matrix<double>b);
    bool degenerate(Matrix<double> vec, Matrix<double> u);
public:
    SimplexMethod(const std::vector<double> A1, const  std::vector<double> b1, const std::vector<double>c1, const  int M1, const  int N1);
    std::vector<double> run();
};

