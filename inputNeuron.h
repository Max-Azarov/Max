#ifndef INPUTNEURON_H
#define INPUTNEURON_H

#include <iostream>

#include "neuron.h"

class InputNeuron : public Neuron
{
private:


public:
    virtual ~InputNeuron() override;
    //void info() override { std::cout << "InputNeuron" << std::endl; }

    void calculateInCalculateOut() override;
    void calculateDelta(Parameters &) override {}
};

#endif // INPUTNEURON_H
