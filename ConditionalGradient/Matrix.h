#pragma once

#include <vector>

using namespace std;


const double MOD = 10000000000000000000;
template<typename T>class Matrix;
template<typename T> Matrix<T> operator*(Matrix<T> one, Matrix<T> two) {
    vector<double>new_matrix(one.lines * two.columns);
    for (int z = 0; z < two.columns; z++) {
        for (int j = 0; j < one.lines; j++) {
            for (int i = 0; i < one.columns; i++) {
                new_matrix[two.columns * j + z] += one.getValue(i, j) * two.getValue(z, i);
            }
        }
    }
    for (int i = 0; i < new_matrix.size(); i++) {
        new_matrix[i] = round(new_matrix[i] * MOD) / MOD;
    }
    return Matrix<T>(new_matrix, one.getLines(), two.getColumns());
};
template<typename T>Matrix<T> operator*(Matrix<T> one, const double val) {
    vector<double>new_matrix(one.lines * one.columns);
    for (int i = 0; i < new_matrix.size(); i++) {
        new_matrix[i] = one.getVector()[i] * val;
    }
    return Matrix<T>(new_matrix, one.lines, one.columns);
};
template<typename T>
class Matrix {
private:
    vector<T> a_i_j;
    int columns;
    int lines;
public:
    Matrix();
    Matrix(int lines, int columns);
    Matrix(vector<T> nums, int n, int m);
    void setMatrix(vector<T> nums, int n, int m);
    vector<T>getVector();
    T getValue(int col, int lin);
    void setValue(T a, int col, int lin);
    int getColumns();
    int getLines();
    double determinant();
    friend Matrix operator-<T>(Matrix one, Matrix two);
    friend Matrix operator*<T>(Matrix one, Matrix two);
    friend Matrix operator+<T>(Matrix one, Matrix two);
    friend Matrix operator*<T>( Matrix one, const double val);
    Matrix Tr();
    Matrix Arange(vector<int> range_x, vector<int> range_y);
    Matrix Samp(vector<int> index_x, vector<int> index_y);
    Matrix multiply(double k);
};



template<typename T> Matrix<T>::Matrix() {
    columns = 0;
    lines = 0;
    a_i_j = {};
}
template<typename T> Matrix<T>::Matrix(int lines, int columns) {
    this->lines = lines;
    this->columns = columns;
    vector<T> nums(lines * columns);
    a_i_j = nums;
}
template<typename T> Matrix<T>::Matrix(vector<T> nums, int n, int m) {

    this->a_i_j = nums;
    this->lines = n;
    this->columns = m;

}
template<typename T> Matrix<T> operator-(Matrix<T> one, Matrix<T> two) {
    vector<T> res(one.columns * one.lines);
    Matrix<T> res_matrix = Matrix<T>(res, one.lines, one.getColumns());
    for (int i = 0; i < one.columns; i++) {
        for (int j = 0; j < one.lines; j++) {
            res_matrix.setValue(one.getValue(i, j) - two.getValue(i, j), i, j);
        }
    }
    for (int i = 0; i < one.columns; i++) {
        for (int j = 0; j < one.lines; j++) {
            res_matrix.setValue(round(res_matrix.getValue(i, j) * MOD) / MOD, i, j);
        }
    }
    return res_matrix;
}
template<typename T> Matrix<T> operator+(Matrix<T> one, Matrix<T> two) {
    vector<T> res(one.columns * one.lines);
    Matrix<T> res_matrix = Matrix<T>(res, one.lines, one.getColumns());
    for (int i = 0; i < one.columns; i++) {
        for (int j = 0; j < one.lines; j++) {
            res_matrix.setValue(one.getValue(i, j) + two.getValue(i, j), i, j);
        }
    }
    for (int i = 0; i < one.columns; i++) {
        for (int j = 0; j < one.lines; j++) {
            res_matrix.setValue(round(res_matrix.getValue(i, j) * MOD) / MOD, i, j);
        }
    }
    return res_matrix;
};
template<typename T> void Matrix<T>::setMatrix(vector<T> nums, int n, int m) {
    a_i_j = nums;
    lines = n;
    columns = m;
}
template<typename T> vector<T> Matrix<T>::getVector()
{
    return a_i_j;
}
template<typename T> T Matrix<T>::getValue(int x, int y) {
    return a_i_j[y * columns + x];
}
template<typename T> void Matrix<T>::setValue(T a, int x, int y) {
    a_i_j[y * columns + x] = a;
}
template<typename T> int Matrix<T>::getColumns() {
    return columns;
}
template<typename T> int Matrix<T>::getLines() {
    return lines;
}
template<typename T> double Matrix<T>::determinant() {
    for (int z = 0; z < columns; z++) {
        double k = getValue(z, z);
        for (int j = 1 + z; j < lines; j++) {
            double c = getValue(z, j);
            for (int i = z; i < columns; i++) {
                setValue(getValue(i, j) - getValue(i, z) * c / k, i, j);
            }
        }
    }
    double sum = getValue(0, 0);
    for (int i = 1; i < columns; i++) {
        sum *= getValue(i, i);
    }
    return sum;
}
template<typename T> Matrix<T> Matrix<T>::Tr() {
    vector<T> res(columns * lines);
    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < lines; j++) {
            res[i * lines + j] = a_i_j[j * columns + i];
        }
    }
    return Matrix(res, columns, lines);
}
template<typename T> Matrix<T> Matrix<T>::Arange(vector<int> range_x, vector<int> range_y) {
    vector<T> res((range_x[1] - range_x[0] + 1) * (range_y[1] - range_y[0] + 1));
    for (int j = range_y[0]; j <= range_y[1]; j++) {
        for (int i = range_x[0]; i <= range_x[1]; i++) {
            res[(j - range_y[0]) * (range_x[1] - range_x[0] + 1) + (i - range_x[0])] = getValue(i, j);
        }
    }
    return Matrix(res, range_y[1] - range_y[0] + 1, range_x[1] - range_x[0] + 1);
}
template<typename T> Matrix<T> Matrix<T>::Samp(vector<int> index_x, vector<int> index_y) {
    vector<T> res(index_x.size() * index_y.size());
    for (int j = 0; j < index_y.size(); j++) {
        for (int i = 0; i < index_x.size(); i++) {
            res[j * index_x.size() + i] = getValue(index_x[i], index_y[j]);
        }
    }
    return Matrix(res, index_y.size(), index_x.size());
}
template<typename T> Matrix<T> Matrix<T>::multiply(double k) {
    vector<T> new_vec = a_i_j;
    for (int i = 0; i < columns * lines; i++) {
        new_vec[i] = new_vec[i] * k;
    }
    return Matrix(new_vec, lines, columns);
}

