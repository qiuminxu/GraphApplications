#include <iostream>
#include <exception>

#include <cuda.h>

int main(int, char **)
{
	//Get CUDA device information.
	int nrDevices = 0;

	cudaGetDeviceCount(&nrDevices);

	if (nrDevices <= 0)
	{
		std::cerr << "No available CUDA devices detected!" << std::endl;
		return -1;
	}

	std::cerr << "We have " << nrDevices << " CUDA device" << (nrDevices > 1 ? "s" : "") << ":" << std::endl;

	for (int i = 0; i < nrDevices; ++i)
	{
		cudaDeviceProp prop;

		cudaGetDeviceProperties(&prop, i);

		std::cerr << i << ": " << prop.name << ", "
			<< "(compute capability " << prop.major << "." << prop.minor << "), "
			<< "global memory " << prop.totalGlobalMem/(1024*1024) << "MiB, "
			<< "const memory " << prop.totalConstMem/1024 << "kiB, "
			<< "shared memory " << prop.sharedMemPerBlock/1024 << "kiB, "
			<< prop.multiProcessorCount << " multiprocessors at " << prop.clockRate/1000 << "MHz, "
			<< "max threads " << prop.maxThreadsPerBlock << ", "
			<< "max block size " << prop.maxThreadsDim[0] << "x" << prop.maxThreadsDim[1] << "x" << prop.maxThreadsDim[2] << ", "
			<< "max grid size " << prop.maxGridSize[0] << "x" << prop.maxGridSize[1] << "x" << prop.maxGridSize[2] << ", "
			<< "warp size " << prop.warpSize << "." << std::endl;
	}
	
	return 0;
}

