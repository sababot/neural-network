neuralnet: main.o layers.o activation.o
	g++ main.o layers.o activation.o -o neuralnet

main.o: main.cpp
	g++ -c main.cpp

layers.o: src/neural_network/src/layers.cpp src/neural_network/include/layers.h
	g++ -c src/neural_network/src/layers.cpp

activation.o: src/neural_network/src/activation.cpp src/neural_network/include/activation.h
	g++ -c src/neural_network/src/activation.cpp
