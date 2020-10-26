#ifndef HIDDENNEURON_H
#define HIDDENNEURON_H

#include <iostream>

#include "neuron.h"

class HiddenNeuron : public Neuron
{
private:

public:
    virtual ~HiddenNeuron() override;
    HiddenNeuron();
    //void info() override { std::cout << "HiddenNeuron" << std::endl; }

};

#endif // HIDDENNEURON_H
