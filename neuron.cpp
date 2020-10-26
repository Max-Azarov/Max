#include "neuron.h"

#include <cmath>

Neuron::Neuron() {
    static size_t id = 0;
    m_id = id;
    //std::cout << "Create Neuron " << m_id << std::endl;
    id++;
    m_delta = 0;
    m_in = 0;
    m_out = 0;
    m_status = ON;
    m_typeActivation = SIGMOID;
}

void Neuron::activationFunction()
{
    double out = 0.0;
    switch (m_typeActivation) {
    case(SIGMOID) :
        // sigmoid
        out = 1.0 / (1.0 + exp(-m_in));
        break;
    case(RELU) :
        // ReLu
        if (m_in > 0) out = m_in;
        else out = 0.1 * m_in;
        break;
    default:;
    }
    setOut(out);
}

double Neuron::getDerivative()
{
    switch (m_typeActivation) {
    case(SIGMOID) :
        // derived from a sigmoid
        return (1.0 - m_out) * m_out;
    case(RELU) :
        // ReLu
        if (m_in > 0) return 1.0;
        else return 0.1;
    default:;
    }
    return 0;
}

void Neuron::calculateInCalculateOut() {
    double sum = 0;
    for (auto it = std::begin(v_synapseIn); it != std::end(v_synapseIn); ++it) {
        sum += (*it)->getOut();
    }
    m_in = sum;
    activationFunction();
    passToSynapse();
}

void Neuron::passToSynapse() {
    for (auto it = std::begin(v_synapseOut); it != std::end(v_synapseOut); ++it) {
        (*it)->setInCalculateOut(m_out);
    }
}

void Neuron::setOut(double out) {
    m_out = out;
    passToSynapse();
}

void Neuron::calculateDelta(Parameters & parameters)
{
    if (m_status == ON) {
        double dTempSum = 0;
        for (auto it = std::begin(v_synapseOut); it != std::end(v_synapseOut); ++it) {
            dTempSum += (*it)->getWeightMultipliedDelta();
        }
        m_delta = getDerivative() * dTempSum;
        for (auto it = std::begin(v_synapseIn); it != std::end(v_synapseIn); ++it) {
            (*it)->updateWeight(m_delta, parameters);
        }
    }
}
