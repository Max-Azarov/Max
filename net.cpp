#include "net.h"
#include "factory.h"

#include <iostream>
#include <fstream>
#include <sstream>

Net::~Net() {}

Net::Net(const std::vector<size_t> & vNeuron,
         const std::vector<size_t> & vSynapse,
         bool newSynapseWeights,
         double e, double a,
         bool randomizeE, bool randomizeA
         )
        : 
          b_newSynapseWeights(newSynapseWeights),
          m_parameters(e, a, randomizeE, randomizeA),
          m_training(this)
{
    initNet(vNeuron, vSynapse);
}

Net::Net(const std::vector<size_t> & vNeuron,
         bool newSynapseWeights,
         double e, double a
         )
        : b_newSynapseWeights(newSynapseWeights),
          m_parameters(e, a),
          m_training(this)
{
    std::vector<size_t> vSynapse( vNeuron.size()-1, 1 );
    initNet(vNeuron, vSynapse);
}

void Net::initNet(const std::vector<size_t> & vNeuron, const std::vector<size_t> & vSynapse)
{
	m_fileName = "weights.bin";
	createFile(m_fileName, b_newSynapseWeights);
	// Проверка размеров vNeuron, vSynapse
    if ( vNeuron.size() < 3 ) {
        std::cerr << "Size of vector vNeuron must be larger than 2" << std::endl;
        exit(EXIT_FAILURE);
    }
    if ( vNeuron.size()-1 != vSynapse.size() ) {
        std::cerr << "Size of vector vNeuron must have 1 element more than vSynapse" << std::endl;
        exit(EXIT_FAILURE);
    }

    BiasNeuronFactory biasFactory;
    HiddenNeuronFactory hiddenFactory;
    InputNeuronFactory inputFactory;
    OutputNeuronFactory outputFactory;

    SynapseFactory synapseFactory;

    m_numberOfLastLayer = vNeuron.size() - 1;
    m_countOfInputs = vNeuron[0];
    m_countOfOutputs = vNeuron[m_numberOfLastLayer];
    m_countOfLayers = vNeuron.size(); // количество всех слоев

    // Создание нейронов входного слоя
    for (size_t i = 0; i < m_countOfInputs; ++i) {
        addNeuron(inputFactory.createNeuron());
    }
    addNeuron(biasFactory.createNeuron()); // Добавляем нейрон смещения

    // Создание нейронов скрытых слоев
    for (size_t i = 1; i < m_numberOfLastLayer; ++i) {
        addLayer();
        for (size_t j = 0; j < vNeuron[i]; ++j) {
            addNeuron(hiddenFactory.createNeuron());
        }
        addNeuron(biasFactory.createNeuron()); // Добавляем нейрон смещения
    }

    // Создание нейронов выходного слоя
    addLayer();
    for (size_t i = 0; i < m_countOfOutputs; ++i) {
        addNeuron(outputFactory.createNeuron());
    }

    // Вектор синапсов должен содержать на 1 меньше элементов, чем вектор нейронов
    // Вектор синапсов описывает на сколько частей делится межнейронное пространство
    // В общем случае в векторе содержаться только значения равные 1

    // Создание синапсов и привязка их к входам и выходам нейронов
    for (size_t x = 0; x < vSynapse.size(); ++x) {
        size_t countOfParts = vSynapse[x];
        size_t temp1 = vNeuron[x] / countOfParts;
        size_t temp2 = vNeuron[x+1] / countOfParts;
        for (size_t z = 0; z < countOfParts - 1; ++z) {
            for (size_t i = z*temp1; i < (z+1)*temp1; ++i) {
                for (size_t j = z*temp2; j < (z+1)*temp2; ++j) {
                    addSynapse(synapseFactory.createSynapse());
                    auto ptr = std::prev(std::end(synapse));
                    neuron[x][i]->attachOutputSynapse(ptr->get());
                    neuron[x+1][j]->attachInputSynapse(ptr->get());
                }
            }
        }
        // Последняя часть может содержать больше синапсов, чем предыдущие при нецелочисленном результате от деления на части
        for (size_t i = temp1*(countOfParts-1); i < vNeuron[x]; ++i) {
            for (size_t j = temp2*(countOfParts-1); j < vNeuron[x+1]; ++j) {
                addSynapse(synapseFactory.createSynapse());
                auto ptr = std::prev(std::end(synapse));
                neuron[x][i]->attachOutputSynapse(ptr->get());
                neuron[x+1][j]->attachInputSynapse(ptr->get());
            }
        }
        // Синапсы нейрона смещения
        for (size_t i = 0; i < vNeuron[x+1]; ++i) {
            addSynapse(synapseFactory.createSynapse());
            auto ptr = std::prev(std::end(synapse));
            neuron[x][vNeuron[x]]->attachOutputSynapse(ptr->get());
            neuron[x+1][i]->attachInputSynapse(ptr->get());
        }
    }

    // Инициализируем дополнительные параметры синапсов
    // Для конвертации из double в string при записи весов в файл и обратно через строковый поток
    std::fstream weightsf; // файловый поток
    //Записываем новые случайные веса синапсов в файл, если это требуется
    if (b_newSynapseWeights) {
        weightsf.open(m_fileName, std::ios::out | std::ios::trunc | std::ios::binary);
        if (!weightsf)
        {
            std::cerr << "\"" << m_fileName << "\" could not be opened!" << std::endl;
            exit(EXIT_FAILURE);
        }
        for (size_t i = 0; i < neuron.size(); ++i) {
            for (size_t j = 0; j < neuron[i].size(); ++j) {
                for (size_t k = 0; k < neuron[i][j]->getOutputSynapse().size(); ++k) {
                    weightsf << m_parameters.getRandom(-1.0, 1.0) << " ";
                }
            }
        }
        weightsf.close();
    }

    // Считываем из файла и устанавливаем веса синапсов
    weightsf.open(m_fileName, std::ios::in | std::ios::binary);
    if (!weightsf)
    {
        std::cerr << "\"" << m_fileName << "\" could not be opened!" << std::endl;
        exit(EXIT_FAILURE);
    }
    double dTemp;
    for (size_t i = 0; i < neuron.size(); ++i) {
        for (size_t j = 0; j < neuron[i].size(); ++j) {
            for ( auto  it =    std::begin(neuron[i][j]->getOutputSynapse());
                        it !=   std::end(neuron[i][j]->getOutputSynapse());
                        ++it)
            {
                if (weightsf >> dTemp) {
                    (*it)->setWeight(dTemp);
                }
                else {
                    std::cerr << "Error when trying to read the file \"" << m_fileName << "\"" << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    weightsf.close();
}

void Net::addNeuron(std::unique_ptr<Neuron> pNeuron) {
    if (!neuron.empty()) {
        size_t vSize = neuron.size();
        neuron.at(vSize - 1).push_back(std::move(pNeuron));
    }
    else {
        addLayer();
        addNeuron(std::move(pNeuron));
    }
}

void Net::addSynapse(std::unique_ptr<Synapse> pSynapse) {
    synapse.push_back(std::move(pSynapse));
}

std::vector<double>& Net::getOut(std::vector<double>& out) const {
    if ( getCountOfOutputs() != out.size() ) {
        std::cerr << "The number of outputs in the training set does not match!" << std::endl;
        exit(EXIT_FAILURE);
    }
    auto outIt = std::begin(out);
    for ( auto it = std::begin(getOutputNeurons());
          it != std::end(getOutputNeurons());
          ++it )
    {
        (*outIt) = (*it)->getOut();
        ++outIt;
    }
    return out;
}

void Net::createFile(const std::string & fileName, bool clearFile) {
    std::fstream file; // файловый поток
    std::ios_base::openmode mode {std::ios::in};
    if (clearFile) mode = std::ios::out | std::ios::trunc | std::ios::binary;

    file.open(fileName, mode);
    if (!file) {
        std::cerr << "\"" << fileName << "\" could not be opened!" << std::endl;
        exit(EXIT_FAILURE);
    }
    file.close();
}




































