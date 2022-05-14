neuralnet: main.o neural_network.o
	g++ main.o neural_network.o -o neuralnet

main.o: main.cpp
	g++ -c main.cpp

neural_network.o: src/neural_network/src/neural_network.cpp src/neural_network/include/neural_network.h
	g++ -c src/neural_network/src/neural_network.cpp
