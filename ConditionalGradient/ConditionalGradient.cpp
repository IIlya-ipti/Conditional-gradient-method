
#include "Matrix.h"
#include "SimplexMethod.h"
#include "GradMethods.h"





/*
    constraint
*/
int M = 3;
int N = 7;

const std::vector<double> A{
         -1,1,0,0,1,0,0,
         0,0,1,-1,0,-1,0,
         2,-2,5,-5,0,0,-1

};
const std::vector<double> b{
    0.1,
    0.1,
    0
};
 
/*
    functions for opt
*/
class f2 : public Func {
public:
    double c_val = 1009000;
    double f(std::vector<double> x) {
        return x[0] * x[0] + x[1] * x[1] + 2 * std::sin((x[0] - x[1]) / 2);;
    }
    double f_original(std::vector<double> x) {
        ;
    }

    Matrix<double>* H(std::vector<double> x) {
        Matrix<double>* Hm = new Matrix<double>(2, 2);
        Hm->setValue(2 - 0.5 * std::sin((x[0] - x[1]) / 2), 0, 0);
        Hm->setValue(0.5 * std::sin((x[0] - x[1]) / 2), 1, 0);
        Hm->setValue(0.5 * std::sin((x[0] - x[1]) / 2), 0, 1);
        Hm->setValue(2 - 0.5 * std::sin((x[0] - x[1]) / 2), 1, 1);
        return Hm;
    }
    Matrix<double> grad(std::vector<double> x) {
        Matrix<double> gr_M = Matrix<double>(2, 1);
        gr_M.setValue(2 * x[0] + std::cos((x[0] - x[1]) / 2), 0, 0);
        gr_M.setValue(2 * x[1] - std::cos((x[0] - x[1]) / 2), 0, 1);
        return gr_M;
    }
    void iter() {
        c_val *= 2;
    };
    double  getVal_C() {
        return c_val;
    }

};

class f3 : public Func {
public:
    double c_val = 1.1;
    double f(std::vector<double> x) {
        return x[0] * x[0] + x[1] * x[1] + 2 * sin((x[0] - x[1]) / 2) + c_val * (
            std::pow(std::max(0.0, g1(x)), 2) +
            std::pow(std::max(0.0, g2(x)), 2) +
            std::pow(std::max(0.0, g3(x)), 2)
            );
    }
    double f_original(std::vector<double> x) {
        ;
    }
    double g1(std::vector<double> x) {
        const double eps = 0.0001;
        return eps - x[1] + b[0];
    }

    double g2(std::vector<double> x) {
        const double eps = 0.0001;
        return eps - x[0] - b[1];
    }
    double g3(std::vector<double> x) {
        const double eps = 0.0001;
        return eps - 2 * x[1] - 5 * x[1];
    }
    Matrix<double>* H(std::vector<double> x) {
        Matrix<double>* Hm = new Matrix<double>(2, 2);
        Hm->setValue(2 - 0.5 * std::sin((x[0] - x[1]) / 2), 0, 0);
        Hm->setValue(0.5 * std::sin((x[0] - x[1]) / 2), 1, 0);
        Hm->setValue(0.5 * std::sin((x[0] - x[1]) / 2), 0, 1);
        Hm->setValue(2 - 0.5 * std::sin((x[0] - x[1]) / 2), 1, 1);
        return Hm;
    }
    Matrix<double> grad(std::vector<double> x) {
        Matrix<double> gr_M = Matrix<double>(2, 1);
        gr_M.setValue(2 * x[0] + std::cos((x[0] - x[1]) / 2) + 2 * c_val *(
            std::max(0.0, g1(x))  *  0,
            std::max(0.0, g2(x))  * -1,
            std::max(0.0, g3(x))  * -2
            ), 0, 0);
        gr_M.setValue(2 * x[1] - std::cos((x[0] - x[1]) / 2) + 2 * c_val * (
            std::max(0.0, g1(x)) * -1,
            std::max(0.0, g2(x)) * 0,
            std::max(0.0, g3(x)) * -5), 0, 1);
        return gr_M;
    }
    void iter() {
        c_val *= 2;
    };
    double  getVal_C() {
        return c_val;
    }

};



int main()
{   



    const double EPS = 0.0001;
    const double ALPHA0 = 0.8;
    const double LAMBDA = 0.5;
    const double R = 10;
    const double D = 1;

    
    /*
        for testing
    */
    const double F = -0.3314809;


    const int COUNT_ADD_COORDS = 3;
    const std::set<int> INDEXES_ADD_COORS = {
        0,
        1
    };


    //search start vector
    f3* f = new f3;
    std::vector<double> x;
    int index = 0;
    do{
        x = minGrad1P(f);
        f->iter();
        index++;
        if (index == 100)break;
    } while (std::abs(std::max(std::max(f->g1(x), f->g2(x)), f->g3(x))) > EPS );

    // vector variable
    std::vector<double> xk = x;
    f2* fun = new f2;

    // learning rate
    double alphak = ALPHA0;

    // vector of grad
    std::vector<double> gr;

    // target func for simplex method
    std::vector<double> target_c(COUNT_ADD_COORDS + xk.size() + INDEXES_ADD_COORS.size(),0);

    // solve of simplex
    std::vector<double> solve_;

    // solve additional task
    std::vector<double> yk(xk.size());

    // 
    std::vector<double> sk(xk.size());

    //
    double hk;

    int it = 0;
    do {

        // get grad in xk
        gr = fun->grad(xk).getVector();

        // update target
        int index = 0;
        for (int i = 0; i < gr.size(); i++) {
            target_c[index] = gr[i];
            if (INDEXES_ADD_COORS.count(i) > 0) {
                index++;
                target_c[index] = -gr[i];
            }
            index++;
        }

        // solve simplex 
        SimplexMethod simp = SimplexMethod(A, b, target_c, M, N);
        solve_ = simp.run();
 

        index = 0;
        for (int i = 0; i < xk.size(); i++) {
            yk[i] = solve_[index];
            if (INDEXES_ADD_COORS.count(i) > 0) {
                index++;
                yk[i] -= solve_[index];
            }
            index++;
        }
       
        sk[0] = yk[0] - xk[0];
        sk[1] = yk[1] - xk[1];

        hk = gr[0] * sk[0] + gr[1] * sk[1];

        std::vector<double> x = xk;
        alphak = ALPHA0;
        while (true)
        {
            x[0] = xk[0] + alphak * sk[0];
            x[1] = xk[1] + alphak * sk[1];
            if ((fun->f(x) - fun->f(xk) )> 0.5 * hk * alphak) {
                alphak *= LAMBDA;
            }
            else {
                break;
            }
        }
        xk = x;
        std::cout << "------" << std::endl;
        std::cout << "iter : " << it << std::endl;
        std::cout << "hk = " << hk << std::endl;
        std::cout << "fk = " <<  fun->f(x) << std::endl;
        std::cout << "alphak = " << alphak << std::endl;
        std::cout << "left = " << -hk * LAMBDA/(2*R* std::pow(modul(sk),2)) << std::endl;
        std::cout << "right = " << -hk  / (2 * R * std::pow(modul(sk), 2)) << std::endl;
        std::cout << "delta fx = " << (fun->f(x) - F)*it << " < " <<D/(it + 1.0)<<  std::endl;
        it++;
        alphak = ALPHA0;
        double val = fun->f(x);
    } while (std::abs(hk) > EPS );

}
