#include "biasNeuron.h"

BiasNeuron::~BiasNeuron()
{
   // std::cout << "Delete Bias Neuron "<< m_id << std::endl;
}

void BiasNeuron::calculateInCalculateOut() {
    m_in = m_out;
    passToSynapse();
}
