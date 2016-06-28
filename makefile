main: poissonImageCloning.o pgm.o
	nvcc -std=c++11 -arch=sm_30 -O2 cpp/main.cu poissonImageCloning.o pgm.o -o main

poissonImageCloning.o: cpp/poissonImageCloning.cu header/common.h
	nvcc -std=c++11 -c cpp/poissonImageCloning.cu

pgm.o: cpp/pgm.cpp
	nvcc -std=c++11 -c cpp/pgm.cpp

clean:
	rm *.o

