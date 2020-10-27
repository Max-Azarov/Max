#include <iostream>
#include <vector>

#include "net.h"

int main()
{
	// Создаем сеть: 2 входа, 1 выход, и один скрытый слой с 4мя нейронами. В каждом слое кроме выходного есть дополнительно нейрон смещения. True - присвоить синапсам случайные веса
	Net net({ 2,4,/**/1 }, true); //   /**/Можно добавить сколь угодно много скрытых слоев с любыми количествами нейронов
	std::cout << "run" << std::endl;

	// Тестовый тренировочный сет "XOR"
	std::vector< std::vector<double>> in;
	std::vector< std::vector<double>> idealOut;

	std::vector<double> in1{ 0.0, 0.0 };
	std::vector<double> in2{ 0.0, 1.0 };
	std::vector<double> in3{ 1.0, 0.0 };
	std::vector<double> in4{ 1.0, 1.0 };
	in.push_back(std::move(in1));
	in.push_back(std::move(in2));
	in.push_back(std::move(in3));
	in.push_back(std::move(in4));

	std::vector<double> idealOut1{ 1.0 };
	std::vector<double> idealOut2{ 0.0 };
	std::vector<double> idealOut3{ 0.0 };
	std::vector<double> idealOut4{ 1.0 };
	idealOut.push_back(std::move(idealOut1));
	idealOut.push_back(std::move(idealOut2));
	idealOut.push_back(std::move(idealOut3));
	idealOut.push_back(std::move(idealOut4));

	double sumError = 0;
	for (size_t era = 0; era < 10000; ++era) {
		for (size_t i = 0; i < in.size(); ++i) {
			net.training().forwardPass(in[i]);
			net.training().backprop(idealOut[i]);
			sumError += net.training().getError();
		}
		if (!(era % 1000)) {
			std::cout << era << " " << sumError / in.size() << std::endl;
		}
		sumError = 0;
	}

	// Выводим значения выходов после обучения
	std::cout << "\n" << std::endl;
	for (size_t i = 0; i < in.size(); ++i) {
		net.training().forwardPass(in[i]);
		std::cout << net.getOutputNeurons()[0]->getOut() << std::endl;
	}
	// Сохраняем веса синапсов
	net.training().saveWeightOfSynapses();
	return 0;
}
