all: cpu


cpu: build_cpu run_cpu

gpu: build_gpu run_gpu


build_cpu:
	cd cpu && g++ ./integration.cpp ./integration_alg.cpp -o ./integration_cpu

run_cpu:
	./cpu/integration_cpu

build_gpu:
	cd gpu && g++ -c ./integration.cpp
	cd gpu && nvcc -Wno-deprecated-gpu-targets --gpu-architecture=compute_35 --gpu-code=sm_35,compute_35 -c ./integration_alg.cu
	cd gpu && g++  -o ./integration_gpu ./integration.o ./integration_alg.o -L/usr/local/cuda/lib64 -lcudart

run_gpu:
	./gpu/integration_gpu


clean:
	rm -f ./cpu/integration_cpu
	rm -f ./gpu/integration_gpu
	rm -f -r ./cpu/*.o
	rm -f -r ./gpu/*.o
