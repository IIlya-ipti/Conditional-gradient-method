#pragma once

#include <vector>
#include <cmath>
#include "Matrix.h"


class Func {
private:
    std::vector<double> c{ 1,1 };
public:
    double f1P(double l) {
        std::vector<double> c_dc = c;
        Matrix<double> dc = grad(c) * l;
        c_dc[0] = c[0] - dc.getValue(0, 0);
        c_dc[1] = c[1] - dc.getValue(0, 1);
        return f(c_dc);
    }
    void setC(std::vector<double> c) {
        this->c = c;
    }
    std::vector <double> getC() {
        return c;
    }

    /*
        function for opt
    */
    virtual double f(std::vector<double> x) = 0;
    /*
        gessian of function
    */
    virtual Matrix<double>* H(std::vector<double> x) = 0;
    /*
        gradient of function
    */
    virtual Matrix<double> grad(std::vector<double> x) = 0;
    virtual double getVal_C() {
        return 0.0;
    }

};

int argmin(std::vector<double> vals);

/*
    uniform search method
*/
double minValue1D(double (Func::* ptrfunc)(double), Func* f, double minX, double maxX);

/*
    gradient method of the first order
*/
std::vector<double> minGrad1P(Func* f);

/*
    norm in Euclidean metric
*/
double modul(vector<double> val);