#include "GradMethods.h"





int argmin(std::vector<double> vals) {
    double mn = vals[0];
    int index = 0;
    for (int i = 1; i < vals.size(); i++) {
        if (vals[i] < mn) {
            mn = vals[i];
            index = i;
        }
    }
    return index;
}

double minValue1D(double (Func::* ptrfunc)(double), Func* f, double minX, double maxX) {
    const double eps = 0.00001;

    // number of x
    const int n = 100;

    // array of func value
    std::vector<double> fs(n);

    // step of method
    double step, mn_val = (f->*ptrfunc)(minX);
    int index;

    do {
        step = (maxX - minX) / n;

        for (int i = 1; i <= n; i++) {
            fs[i - 1] = (f->*ptrfunc)(minX + step * i);
        }
        index = argmin(fs);
        minX = minX + step * (index - 1.0);
        maxX = minX + 2 * step;

    } while (abs(minX - maxX) > eps);

    return (minX + maxX) / 2;
}

std::vector<double> minGrad1P(Func* f) {

    // constants
    double eps = 0.00001;

    // value of alpha k (learning rate)
    double l;

    int steps = 0;

    // start position
    std::vector<double> x = f->getC();


    Matrix<double> dx;
    do {
        steps += 1;
        f->setC(x);

        l = minValue1D(&(Func::f1P), f, 0.00000000000001, 0.99999999999999);
        //l = 0.12 / f->getVal_C();

        dx = f->grad(x) * l;
        x[0] = x[0] - dx.getValue(0, 0);
        x[1] = x[1] - dx.getValue(0, 1);
        if (steps == 50)break;
    } while (std::abs(std::max<double>(std::abs(dx.getValue(0, 0)), std::abs(dx.getValue(0, 1)))) > eps);

    return x;
}


double modul(std::vector<double> val) {
    double sm = 0;
    for (int i = 0; i < val.size(); i++) {
        sm += val[i] * val[i];
    }
    return std::pow(sm, 0.5);
}