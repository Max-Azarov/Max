#ifndef NET_H
#define NET_H

#include <cassert>
#include <vector>
#include <memory>
#include <list>

#include "_neuronet.h"
#include "training.h"

class Net
{
private:
    size_t m_countOfInputs;
    size_t m_countOfOutputs;
    size_t m_countOfLayers;
    size_t m_numberOfLastLayer;
    std::string m_fileName;

    bool b_newSynapseWeights = true;

    std::vector<std::vector<std::unique_ptr<Neuron>>> neuron;   // Владелец нейронов
    std::list<std::unique_ptr<Synapse>> synapse;                // Владелец синапсов

    Parameters m_parameters;
    Training m_training;

private:
    void addLayer() { neuron.push_back(std::vector<std::unique_ptr<Neuron>>()); }
    void addNeuron(std::unique_ptr<Neuron>);
    void addSynapse(std::unique_ptr<Synapse>);

public:
    // В конструктор передается массив, описывающий сеть нейронов,
    // начиная со входного слоя, количество нейронов в каждом слое,
    // вплоть до выходного слоя
    // Массив синапсов описывает, на сколько частей делится межнейронное
    // пространство между каждыми слоями. Должно быть на 1 элемент меньше, чем в
    // массиве нейронов
    // Если нет разрывов и каждый нейрон слоя соединей с каждым нейроном следущего
    // слоя, то массив синапсов должени содержать только 1.
    Net(const std::vector<size_t> & vNeuron,
        const std::vector<size_t> & vSynapse,
        bool newSynapseWeights = true,
        double e = 0.3, double a = 0.1,
        bool randomizeE = true, bool m_randomizeA = true
        );
    Net(const std::vector<size_t> & vNeuron,
        bool newSynapseWeights,
        double e = 0.1, double a = 0.1
        );

    ~Net();

    void initNet(const std::vector<size_t> & vNeuron, const std::vector<size_t> & vSynapse);
    void createFile(const std::string & fileName, bool clearFile);

    size_t getCountOfInputs() const { return m_countOfInputs; }
    size_t getCountOfOutputs() const { return m_countOfOutputs; }
    size_t getCountOfLayers() const { return m_countOfLayers; }
    size_t getNumberOfLastLayer() const { return m_numberOfLastLayer; }

    const std::vector<std::vector<std::unique_ptr<Neuron>>> & getNeuron() const { return neuron; }
    const std::vector<std::unique_ptr<Neuron>> & getInputNeurons() const { return neuron[0]; }
    const std::vector<std::unique_ptr<Neuron>> & getOutputNeurons() const { return neuron[m_numberOfLastLayer]; }
    const std::list<std::unique_ptr<Synapse>> & getSynapses() const { return synapse; }
    std::vector<double>& getOut(std::vector<double>&) const;

    Parameters & parameters() { return m_parameters; }
    Training & training() { return m_training; }
};

#endif // NET_H
