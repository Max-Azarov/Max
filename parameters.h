#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <ctime>
#include <cstdlib>
#include <iostream>

class Parameters
{
private:
    double m_e; // learning rate
    double m_a; // moment
    bool m_randomizeE; // применение случайности в возврате скорости
    bool m_randomizeA; // применение случайности в возврате момента

public:
    Parameters() {}
    Parameters(double e = 0.1, double a = 0.1, bool randomizeE = true, bool randomizeA = true);

    void setRandomizeE(bool randomizeE) { m_randomizeE = randomizeE; }
    void setE(double e, bool randomizeE) { m_e = e; m_randomizeE = randomizeE; }
    void setE(double e) { m_e = e; }
    double getE();

    void setRandomizeA(bool randomizeA) { m_randomizeA = randomizeA; }
    void setA(double a, bool randomizeA) { m_a = a; m_randomizeA = randomizeA; }
    void setA(double a) { m_a = a; }
    double getA();

    template <typename T>
    double getRandom(T firstBoundaryValue, T secondBoundaryValue) {
        static bool bsrand = false;
        if( !bsrand ) {
            srand(static_cast<size_t>(time(nullptr)));
            bsrand = true;
        }
        double result = firstBoundaryValue > secondBoundaryValue ?
                        secondBoundaryValue + (firstBoundaryValue - secondBoundaryValue) * static_cast<double>(rand()) / RAND_MAX :
                        firstBoundaryValue + (secondBoundaryValue - firstBoundaryValue) * static_cast<double>(rand()) / RAND_MAX;
        return result;
    }
};

#endif // PARAMETERS_H
