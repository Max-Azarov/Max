#include "inputNeuron.h"

InputNeuron::~InputNeuron() {
    //std::cout << "Delete InputNeuron "<< m_id << std::endl;
}

void InputNeuron::calculateInCalculateOut() {
    m_out = m_in;
    passToSynapse();
}
