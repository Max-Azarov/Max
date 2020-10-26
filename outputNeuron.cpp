#include "outputNeuron.h"

OutputNeuron::~OutputNeuron()
{
    //std::cout << "Delete OutputNeuron "<< m_id << std::endl;
}
/*
void OutputNeuron::setInCalculateOut(double in)
{
    if (m_status == ON) {
        m_in = in;
        activationFunction();
    }
}
*/
void OutputNeuron::calculateDelta(Parameters & parameters)
{
    if (m_status == ON) {
        m_delta = (m_idealOut - m_out) * getDerivative();
        for (auto it = std::begin(v_synapseIn); it != std::end(v_synapseIn); ++it) {
            (*it)->updateWeight(m_delta, parameters);
        }
    }
}

void OutputNeuron::calculateInCalculateOut() {
    double sum = 0;
    for (auto it = std::begin(v_synapseIn); it != std::end(v_synapseIn); ++it) {
        sum += (*it)->getOut();
    }
    m_in = sum;
    activationFunction();
}
