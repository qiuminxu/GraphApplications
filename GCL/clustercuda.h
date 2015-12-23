#ifndef CLUSTER_CLUSTER_CUDA_H
#define CLUSTER_CLUSTER_CUDA_H

#include <vector>

#include <thrust/device_vector.h>

#include "graph.h"
#include "cluster.h"
#include "vis.h"

namespace clu
{

class ClusterCUDA : public Cluster
{
	public:
		ClusterCUDA(const int & = 256);
		~ClusterCUDA();
		
		std::vector<int> cluster(const Graph &, const double & = 1.0, Drawer * = 0) const;
		
	private:
		void match(int * const, int * const, const int &, const long &) const;
		void printDeviceVector(const thrust::device_vector<int> &) const;
		void printDeviceVector(const thrust::device_vector<int2> &) const;
		Graph createGraph(const thrust::device_vector<int2> &,
					const thrust::device_vector<int2> &,
					const thrust::device_vector<int> &,
					const thrust::device_vector<float2> &,
					const int &) const;
		
		const int blockSize;
};

}

#endif

