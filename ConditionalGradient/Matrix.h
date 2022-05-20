#pragma once

#include "includes.h"


const double MOD = 10000000000000000000;


/*
    operations  matrix
*/
template<typename T>
class Matrix;
template<class T> Matrix<T> operator*(Matrix<T> one, Matrix<T> two);
template<typename T>Matrix<T> operator*(Matrix<T> one, const double val);
template<typename T> Matrix<T> operator-(Matrix<T> one, Matrix<T> two);
template<typename T> Matrix<T> operator+(Matrix<T> one, Matrix<T> two);



template<typename T>
class Matrix {
private:
    std::vector<T> a_i_j;
    int columns;
    int lines;
public:
    Matrix();
    Matrix(int lines, int columns);
    Matrix(std::vector<T> nums, int n, int m);
    void setMatrix(std::vector<T> nums, int n, int m);
    std::vector<T>getVector();
    T getValue(int col, int lin);
    void setValue(T a, int col, int lin);
    int getColumns();
    int getLines();
    double determinant();
    friend Matrix operator-<T>(Matrix one, Matrix two);
    friend Matrix operator*<T>(Matrix one, Matrix two);
    friend Matrix operator+<T>(Matrix one, Matrix two);
    friend Matrix operator*<T>(Matrix one, const double val);
    Matrix Tr();
    Matrix Arange(std::vector<int> range_x,std::vector<int> range_y);
    Matrix Samp(std::vector<int> index_x, std::vector<int> index_y);
    Matrix multiply(double k);
};



