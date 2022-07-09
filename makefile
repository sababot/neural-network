neuralnet: main.o layers.o activation.o import_data.o
	g++ main.o layers.o activation.o import_data.o -o neuralnet

main.o: main.cpp
	g++ -c main.cpp

layers.o: src/neural_network/src/layers.cpp src/neural_network/include/layers.h
	g++ -c src/neural_network/src/layers.cpp

activation.o: src/neural_network/src/activation.cpp src/neural_network/include/activation.h
	g++ -c src/neural_network/src/activation.cpp

import_data.o: src/utils/src/import_data.cpp src/utils/include/import_data.h
	g++ -c src/utils/src/import_data.cpp

clean:
	rm *.o
	rm neuralnet