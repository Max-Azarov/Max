#ifndef BIASNEURON_H
#define BIASNEURON_H

#include "neuron.h"

class BiasNeuron : public Neuron
{
private:
    void setOut(double) {}

public:
    //void info() override;
    virtual ~BiasNeuron() override;

    BiasNeuron() { m_out = 1.0; }

    void calculateInCalculateOut() override;
    void calculateDelta(Parameters &) override {}
};

#endif // BIASNEURON_H
