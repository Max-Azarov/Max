#ifndef SYNAPSE_H
#define SYNAPSE_H

#include <iostream>

#include "parameters.h"

enum STATUS {
    OFF,
    ON
};

class Neuron;

class Synapse
{
protected:
    double m_weight;
    double m_prevWeight;
    double m_dWeight = 0;
    double m_in;
    double m_out;
    double m_grad;
    double m_delta;
    size_t m_id;

    STATUS m_status;

public:
    Synapse();

    size_t getID() const { return m_id; }

    virtual void info() { std::cout << "Synapse "
                            << "\tweight=" << m_weight
                            //<< "\tout=" << m_out
                            << "\tgrad=" << m_grad
                            << "\tdWeight=" << m_dWeight
                            << std::endl; }

    virtual ~Synapse() { //
        //std::cout << "Delete Synapse " << m_id << std::endl;
    }

    void setWeight(double weight) { m_weight = weight; }
    double getWeight() const { return m_weight; }

    double getDWeight() const { return m_dWeight; }

    void setInCalculateOut(double in) { if (m_status == ON) {m_in = in; m_out = m_in * m_weight;} }
    double getIn() const { return m_in; }

    double getOut() const { return m_status == ON ? m_out : 0.0; }
    double getDelta() const { return m_delta; }
    double getGrad() const { return m_grad; }

    virtual void updateWeight(double delta, Parameters & parameters);
    virtual double getWeightMultipliedDelta() { return m_status ? m_prevWeight * m_delta : 0.0; }

    void setStatus(STATUS status) { m_status = status; }
    STATUS getStatus() const { return m_status; }
};

#endif // SYNAPSE_H
