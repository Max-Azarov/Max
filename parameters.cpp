#include "parameters.h"

Parameters::Parameters(double e, double a, bool randomizeE, bool randomizeA) :
    m_e { e },
    m_a { a },
    m_randomizeE { randomizeE },
    m_randomizeA { randomizeA }
{

}

double Parameters::getE()
{
    if (m_randomizeE) {
        return getRandom(m_e/2.0, m_e*2.0);
    }
    return m_e;
}

double Parameters::getA()
{
    if (m_randomizeA) {
        return getRandom(m_a/2.0, m_a*2.0);
    }
    return m_a;
}
