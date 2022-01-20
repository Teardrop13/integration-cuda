all: build run

build:
	g++ -c ./integration.cpp ./integration_alg_cpu.cpp ./generator.cpp
	# nvcc -Wno-deprecated-gpu-targets --gpu-architecture=compute_35 --gpu-code=sm_35,compute_35 -c ./integration_alg_gpu.cu
	# g++  -o ./integration ./integration.o ./integration_alg_cpu.o ./integration_alg_gpu.o ./generator.o -L/usr/local/cuda/lib64 -lcudart
	g++  -o ./integration ./integration.o ./integration_alg_cpu.o ./generator.o

run:
	./integration

clean:
	rm -f ./integration
