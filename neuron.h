#ifndef NEURON_H
#define NEURON_H

#include "parameters.h"
#include "synapse.h"

#include <iostream>
#include <list>

enum TYPE_ACTIVATION
{
    SIGMOID,
    RELU,
    NUM_OF_TYPE_ACTIVATION
};

class Synapse;

class Neuron
{
protected:
    size_t m_id;
    double m_in;
    double m_out;
    double m_delta;
    STATUS m_status;

    TYPE_ACTIVATION m_typeActivation;

    void activationFunction();
    double getDerivative();
    void passToSynapse();

    std::list<Synapse*> v_synapseOut;
    std::list<Synapse*> v_synapseIn;

    void setOut(double out);

public:
    Neuron();

    virtual void info() { std::cout << "Neuron " << " in=" << m_in << "\tout="
                        << m_out << "\tdelta=" << m_delta << "\n"; }
    virtual ~Neuron() {
        //std::cout << "Delete Neuron " << m_id << std::endl;
    }

    void setIn(double in) { m_in = in; }
    double getIn() const { return m_in; }


    double getOut() const { return m_status == ON ? m_out : 0.0; }
    double getDelta() const { return m_status == ON ? m_delta : 0.0; }

    void setStatus(STATUS status) { m_status = status; }
    STATUS getStatus() const { return m_status; }
    
    virtual void calculateInCalculateOut();
    virtual void calculateDelta(Parameters & parameters);

    const std::list<Synapse*> & getInputSynapse() { return v_synapseIn; }
    void attachInputSynapse(Synapse* pSynapse) { v_synapseIn.push_back(pSynapse); }
    void resetInputSynapse() { v_synapseIn.clear(); }
    const std::list<Synapse*> & getOutputSynapse() { return v_synapseOut; }
    void attachOutputSynapse(Synapse* pSynapse) { v_synapseOut.push_back(pSynapse); }
    void resetOutputSynapse() { v_synapseOut.clear(); }

};

#endif // NEURON_H
