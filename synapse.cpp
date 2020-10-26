#include "synapse.h"

Synapse::Synapse()
{
    static size_t id = 0;
    m_id = id;
    //std::cout << "Create Synapse " << m_id << std::endl;
    id++;
    m_status = ON;
}

void Synapse::updateWeight(double delta, Parameters & parameters)
{
    if (m_status == ON) {
        m_delta = delta;
        m_grad = m_delta * m_in;
        m_dWeight = parameters.getE() * m_grad +
                    parameters.getA() * m_dWeight;
        m_prevWeight = m_weight;
        m_weight += m_dWeight;
    }
}
