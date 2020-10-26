#include <string>
#include <sstream>
#include <fstream>

#include "training.h"
#include "net.h"
#include "_neuronet.h"

Training::Training(Net* net)
    : p_Net     { net },
      m_error   { 1.0 }
{
}

void Training::forwardPass(const std::vector<double> & in ) {
    if ( p_Net->getCountOfInputs() != in.size() ) {
        std::cerr << "The number of inputs in the training set does not match!" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Передача значений на входы сети
    size_t count = 0;
    for (   auto itIn = std::begin(p_Net->getInputNeurons());
            itIn != std::prev(std::end(p_Net->getInputNeurons())); // prev, чтобы не присваивать значение входа нейрону смещения
            ++itIn   )
    {
        (*itIn)->setIn(static_cast<double>(in[count]));
        ++count;
    }

    // Прямой проход по сети
    for (auto itLayer = std::begin(p_Net->getNeuron()); itLayer != std::end(p_Net->getNeuron()); ++itLayer) {
        for (auto itNeuron = std::begin(*itLayer); itNeuron != std::end(*itLayer); ++itNeuron) {
            (*itNeuron)->calculateInCalculateOut();
        }
    }
}

double Training::calculateError(const std::vector<double> & idealOut) {
    if ( p_Net->getCountOfOutputs() != idealOut.size() ) {
        std::cerr << "The number of outputs in the training set for calculating the error does not match!" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Считаем ошибку после обучающей выборки
    // MSE   ERROR = ((I1 - A1)^2 + (I2 - A2)^2 + ... + (In - In)^2) / n
    double tempError = 0.0;
    size_t count = 0;
    for (   auto itOut = std::begin(p_Net->getOutputNeurons());
            itOut != std::end(p_Net->getOutputNeurons());
            ++itOut )
    {
        tempError += ((*itOut)->getOut() - idealOut[count])*((*itOut)->getOut() - idealOut[count]);
        ++count;
    }
    m_error = tempError / p_Net->getCountOfOutputs();
    return m_error;
}

void Training::backprop(const std::vector<double> & idealOut) {
    if ( p_Net->getCountOfOutputs() != idealOut.size() ) {
        std::cerr << "The number of outputs in the training set does not match!" << std::endl;
        exit(EXIT_FAILURE);
    }
    // Считаем ошибку
    calculateError(idealOut);

    // Передача идеальных значений в выходные нейроны
    size_t count = 0;
    for (   auto itOut = std::begin(p_Net->getOutputNeurons());
            itOut != std::end(p_Net->getOutputNeurons());
            ++itOut )
    {
        dynamic_cast<OutputNeuron*>(itOut->get())->setIdealOut(idealOut[count] );
        ++count;
    }

    // Распространяем ошибку в обратном направлении
    for (auto itLayer = std::end(p_Net->getNeuron()); itLayer != std::begin(p_Net->getNeuron()); --itLayer) {
        for (auto itNeuron = std::begin(*std::prev(itLayer)); itNeuron != std::end(*std::prev(itLayer)); ++itNeuron) {
            (*itNeuron)->calculateDelta(p_Net->parameters());
        }
    }
}

void Training::saveWeightOfSynapses() {
    // Сохраняем веса синапсов в "weights.bin"
    std::string fileName {"weights.bin"};
    std::fstream weightsf; // файловый поток

    weightsf.open(fileName, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!weightsf)
    {
        std::cerr << "\"" << fileName << "\" could not be opened!" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < p_Net->getNeuron().size(); ++i) {
        for (size_t j = 0; j < p_Net->getNeuron()[i].size(); ++j) {
            for ( auto  it =    std::begin(p_Net->getNeuron()[i][j]->getOutputSynapse());
                        it !=   std::end(p_Net->getNeuron()[i][j]->getOutputSynapse());
                        ++it) {
                weightsf << (*it)->getWeight() << " ";
            }
        }
    }
    weightsf.close();
}
