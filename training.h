#ifndef TRAINING_H
#define TRAINING_H

#include <vector>

class Net;

class Training {
private:
    Net* p_Net;

    double m_error;

public:
    Training(Net*);

    double getError() const { return m_error; }
    void forwardPass(const std::vector<double> & in);
    double calculateError(const std::vector<double> & idealOut);
    void backprop(const std::vector<double> & idealOut);
    void saveWeightOfSynapses();
};

#endif // TRAINING_H
