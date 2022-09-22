neuralnet: main.o layers.o activation.o loss.o optimization.o import_data.o utils.o
	g++ main.o layers.o activation.o loss.o optimization.o import_data.o utils.o -o neuralnet

main.o: main.cpp
	g++ -c main.cpp

layers.o: src/neural_network/src/layers.cpp src/neural_network/include/layers.h
	g++ -c src/neural_network/src/layers.cpp

activation.o: src/neural_network/src/activation.cpp src/neural_network/include/activation.h
	g++ -c src/neural_network/src/activation.cpp

loss.o: src/neural_network/src/loss.cpp src/neural_network/include/loss.h
	g++ -c src/neural_network/src/loss.cpp

optimization.o: src/neural_network/src/optimization.cpp src/neural_network/include/optimization.h
	g++ -c src/neural_network/src/optimization.cpp

import_data.o: src/utils/src/import_data.cpp src/utils/include/import_data.h
	g++ -c src/utils/src/import_data.cpp

utils.o: src/utils/src/utils.cpp src/utils/include/utils.h
	g++ -c src/utils/src/utils.cpp

clean:
	rm *.o
	rm neuralnet
