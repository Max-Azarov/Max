#ifndef FACTORY_H
#define FACTORY_H

#include <memory>

#include "_neuronet.h"

class NeuronFactory {
public:
    virtual std::unique_ptr<Neuron> createNeuron() = 0;
    virtual ~NeuronFactory() {}
};

class BiasNeuronFactory : public NeuronFactory {
public:
    std::unique_ptr<Neuron> createNeuron() override {
        return std::unique_ptr<Neuron>(new BiasNeuron);
    }
};

class HiddenNeuronFactory : public NeuronFactory {
public:
    std::unique_ptr<Neuron> createNeuron() override {
        return std::unique_ptr<Neuron>(new HiddenNeuron);
    }
};

class InputNeuronFactory : public NeuronFactory {
public:
    std::unique_ptr<Neuron> createNeuron() override {
        return std::unique_ptr<Neuron>(new InputNeuron);
    }
};

class OutputNeuronFactory : public NeuronFactory {
public:
    std::unique_ptr<Neuron> createNeuron() override {
        return std::unique_ptr<Neuron>(new OutputNeuron);
    }
};

class SFactory {
public:
    virtual std::unique_ptr<Synapse> createSynapse() = 0;
    virtual ~SFactory() {}
};

class SynapseFactory : public SFactory {
public:
    std::unique_ptr<Synapse> createSynapse() override {
        return std::unique_ptr<Synapse>(new Synapse);
    }
};

#endif // FACTORY_H
