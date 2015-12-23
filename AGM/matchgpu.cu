/*
Copyright 2011, Bas Fagginger Auer.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#include <iostream>
#include <exception>
#include <string>
#include <algorithm>
#include <cassert>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

//CUDA 3.2 does not seem to make definitions for texture types.
#ifndef cudaTextureType1D
#define cudaTextureType1D 0x01
#endif

#include "matchgpu.h"

using namespace std;
using namespace mtc;

__constant__ uint dSelectBarrier = 0x8000000;

GraphMatchingGPU::GraphMatchingGPU(const Graph &_graph, const int &_threadsPerBlock, const unsigned int &_selectBarrier) :
		threadsPerBlock(_threadsPerBlock),
		GraphMatching(_graph),
		selectBarrier(_selectBarrier)
{
	//Allocate memory to store the graph on the device.
	if (cudaMalloc(&dneighbourRanges, sizeof(int2)*graph.neighbourRanges.size()) != cudaSuccess
		|| cudaMalloc(&dneighbours, sizeof(int)*graph.neighbours.size()) != cudaSuccess)
	{
		cerr << "Not enough memory on device to store this graph!" << endl;
		throw exception();
	}

	//Copy graph data to device.
	if (cudaMemcpy(dneighbourRanges, &graph.neighbourRanges[0], sizeof(int2)*graph.neighbourRanges.size(), cudaMemcpyHostToDevice) != cudaSuccess
		|| cudaMemcpy(dneighbours, &graph.neighbours[0], sizeof(int)*graph.neighbours.size(), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cerr << "Unable to transfer graph data to device!" << endl;
		throw exception();
	}

	//Set select barrier.
	if (cudaMemcpyToSymbol(dSelectBarrier, &selectBarrier, sizeof(uint)) != cudaSuccess)
	{
		cerr << "Unable to set selection barrier!" << endl;
		throw exception();
	}
}

GraphMatchingGPU::~GraphMatchingGPU()
{
	//Free all graph data on the GPU.
	cudaFree(dneighbours);
	cudaFree(dneighbourRanges);
}

GraphMatchingGPURandom::GraphMatchingGPURandom(const Graph &_graph, const int &_nrThreads, const unsigned int &_selectBarrier) :
		GraphMatchingGPU(_graph, _nrThreads, _selectBarrier)
{

}

GraphMatchingGPURandom::~GraphMatchingGPURandom()
{

}

GraphMatchingGPURandomMaximal::GraphMatchingGPURandomMaximal(const Graph &_graph, const int &_nrThreads, const unsigned int &_selectBarrier) :
		GraphMatchingGPU(_graph, _nrThreads, _selectBarrier)
{

}

GraphMatchingGPURandomMaximal::~GraphMatchingGPURandomMaximal()
{

}

GraphMatchingGPUWeighted::GraphMatchingGPUWeighted(const Graph &_graph, const int &_nrThreads, const unsigned int &_selectBarrier) :
		GraphMatchingGPU(_graph, _nrThreads, _selectBarrier)
{
	assert(graph.neighbourWeights.size() == graph.neighbours.size());

	//Allocate memory on the device to store the graph weights.
	if (cudaMalloc(&dweights, sizeof(float)*graph.neighbourWeights.size()) != cudaSuccess)
	{
		cerr << "Not enough memory on device to store graph weights!" << endl;
		throw exception();
	}

	//Copy weights.
	if (cudaMemcpy(dweights, &graph.neighbourWeights[0], sizeof(float)*graph.neighbourWeights.size(), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cerr << "Unable to transfer graph weights to device!" << endl;
		throw exception();
	}
}

GraphMatchingGPUWeighted::~GraphMatchingGPUWeighted()
{
	//Free weights.
	cudaFree(dweights);
}

GraphMatchingGPUWeightedMaximal::GraphMatchingGPUWeightedMaximal(const Graph &_graph, const int &_nrThreads, const unsigned int &_selectBarrier) :
		GraphMatchingGPU(_graph, _nrThreads, _selectBarrier)
{
	assert(graph.neighbourWeights.size() == graph.neighbours.size());

	//Allocate memory on the device to store the graph weights.
	if (cudaMalloc(&dweights, sizeof(float)*graph.neighbourWeights.size()) != cudaSuccess)
	{
		cerr << "Not enough memory on device to store graph weights!" << endl;
		throw exception();
	}

	//Copy weights.
	if (cudaMemcpy(dweights, &graph.neighbourWeights[0], sizeof(float)*graph.neighbourWeights.size(), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		cerr << "Unable to transfer graph weights to device!" << endl;
		throw exception();
	}
}

GraphMatchingGPUWeightedMaximal::~GraphMatchingGPUWeightedMaximal()
{
	//Free weights.
	cudaFree(dweights);
}

//==== Kernel variables ====
__device__ int dkeepMatching;

texture<int2, cudaTextureType1D, cudaReadModeElementType> neighbourRangesTexture;
texture<int, cudaTextureType1D, cudaReadModeElementType> neighboursTexture;
texture<float, cudaTextureType1D, cudaReadModeElementType> weightsTexture;

//==== General matching kernels ====
/*
   Match values match[i] have the following interpretation for a vertex i:
   0   = blue,
   1   = red,
   2   = unmatchable (all neighbours of i have been matched),
   3   = reserved,
   >=4 = matched.
*/

//Nothing-up-my-sleeve working constants from SHA-256.
__constant__ const uint dMD5K[64] = {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
				0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
				0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
				0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
				0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
				0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
				0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
				0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

//Rotations from MD5.
__constant__ const uint dMD5R[64] = {7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
				5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
				4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
				6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21};

#define LEFTROTATE(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

__global__ void gSelect(int *match, const int nrVertices, const uint random)
{
	//Determine blue and red groups using MD5 hashing.
	//Based on the Wikipedia MD5 hashing pseudocode (http://en.wikipedia.org/wiki/MD5).
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;

	//Can this vertex still be matched?
	if (match[i] >= 2) return;

	//Start hashing.
	uint h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476;
	uint a = h0, b = h1, c = h2, d = h3, e, f, g = i;

	for (int j = 0; j < 16; ++j)
	{
		f = (b & c) | ((~b) & d);

		e = d;
		d = c;
		c = b;
		b += LEFTROTATE(a + f + dMD5K[j] + g, dMD5R[j]);
		a = e;

		h0 += a;
		h1 += b;
		h2 += c;
		h3 += d;

		g *= random;
	}
	
	match[i] = ((h0 + h1 + h2 + h3) < dSelectBarrier ? 0 : 1);
}

__global__ void gaSelect(int *match, const int nrVertices, const uint random)
{
	//Determine blue and red groups using MD5 hashing.
	//Based on the Wikipedia MD5 hashing pseudocode (http://en.wikipedia.org/wiki/MD5).
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;

	//Can this vertex still be matched?
	if (match[i] >= 2) return;

	//Use atomic operations to indicate whether we are done.
	//atomicCAS(&dkeepMatching, 0, 1);
	dkeepMatching = 1;

	//Start hashing.
	uint h0 = 0x67452301, h1 = 0xefcdab89, h2 = 0x98badcfe, h3 = 0x10325476;
	uint a = h0, b = h1, c = h2, d = h3, e, f, g = i;

	for (int j = 0; j < 16; ++j)
	{
		f = (b & c) | ((~b) & d);

		e = d;
		d = c;
		c = b;
		b += LEFTROTATE(a + f + dMD5K[j] + g, dMD5R[j]);
		a = e;

		h0 += a;
		h1 += b;
		h2 += c;
		h3 += d;

		g *= random;
	}
	
	match[i] = ((h0 + h1 + h2 + h3) < dSelectBarrier ? 0 : 1);
}

__global__ void gMatch(int *match, const int *requests, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;

	const int r = requests[i];

	//Only unmatched vertices make requests.
	if (r == nrVertices + 1)
	{
		//This is vertex without any available neighbours, discard it.
		match[i] = 2;
	}
	else if (r < nrVertices)
	{
		//This vertex has made a valid request.
		if (requests[r] == i)
		{
			//Match the vertices if the request was mutual.
			match[i] = 4 + min(i, r);
		}
	}
}

//==== Random greedy matching kernels ====
__global__ void grRequest(int *requests, const int *match, const int nrVertices)
{
	//Let all blue vertices make requests.
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);

	//Look at all blue vertices and let them make requests.
	if (match[i] == 0)
	{
		int dead = 1;

		for (int j = indices.x; j < indices.y; ++j)
		{
			const int ni = tex1Dfetch(neighboursTexture, j);
			const int nm = match[ni];

			//Do we have an unmatched neighbour?
			if (nm < 4)
			{
				//Is this neighbour red?
				if (nm == 1)
				{
					//Propose to this neighbour.
					requests[i] = ni;
					return;
				}
				
				dead = 0;
			}
		}

		requests[i] = nrVertices + dead;
	}
	else
	{
		//Clear request value.
		requests[i] = nrVertices;
	}
}

__global__ void grRespond(int *requests, const int *match, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);

	//Look at all red vertices.
	if (match[i] == 1)
	{
		//Select first available proposer.
		for (int j = indices.x; j < indices.y; ++j)
		{
			const int ni = tex1Dfetch(neighboursTexture, j);

			//Only respond to blue neighbours.
			if (match[ni] == 0)
			{
				//Avoid data thrashing be only looking at the request value of blue neighbours.
				if (requests[ni] == i)
				{
					requests[i] = ni;
					return;
				}
			}
		}
	}
}

//==== Weighted greedy matching kernels ====
__global__ void gwRequest(int *requests, const int *match, const int nrVertices)
{
	//Let all blue vertices make requests.
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);

	//Look at all blue vertices and let them make requests.
	if (match[i] == 0)
	{
		float maxWeight = -1.0;
		int candidate = nrVertices;
		int dead = 1;

		for (int j = indices.x; j < indices.y; ++j)
		{
			//Only propose to red neighbours.
			const int ni = tex1Dfetch(neighboursTexture, j);
			const int nm = match[ni];

			//Do we have an unmatched neighbour?
			if (nm < 4)
			{
				//Is this neighbour red?
				if (nm == 1)
				{
					//Propose to the heaviest neighbour.
					const float nw = tex1Dfetch(weightsTexture, j);

					if (nw > maxWeight)
					{
						maxWeight = nw;
						candidate = ni;
					}
				}
				
				dead = 0;
			}
		}

		requests[i] = candidate + dead;
	}
	else
	{
		//Clear request value.
		requests[i] = nrVertices;
	}
}

__global__ void gwRespond(int *requests, const int *match, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);

	//Look at all red vertices.
	if (match[i] == 1)
	{
		float maxWeight = -1;
		int candidate = nrVertices;

		//Select heaviest available proposer.
		for (int j = indices.x; j < indices.y; ++j)
		{
			const int ni = tex1Dfetch(neighboursTexture, j);

			//Only respond to blue neighbours.
			if (match[ni] == 0)
			{
				if (requests[ni] == i)
				{
					const float nw = tex1Dfetch(weightsTexture, j);

					if (nw > maxWeight)
					{
						maxWeight = nw;
						candidate = ni;
					}
				}
			}
		}

		if (candidate < nrVertices)
		{
			requests[i] = candidate;
		}
	}
}

void GraphMatchingGPURandom::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2) const
{
	//Creates a greedy random matching on the GPU.
	//Assumes the current matching is empty.

	assert((int)match.size() == graph.nrVertices);
	
	//Setup textures.
	cudaChannelFormatDesc neighbourRangesTextureDesc = cudaCreateChannelDesc<int2>();

	neighbourRangesTexture.addressMode[0] = cudaAddressModeWrap;
	neighbourRangesTexture.filterMode = cudaFilterModePoint;
	neighbourRangesTexture.normalized = false;
	cudaBindTexture(0, neighbourRangesTexture, (void *)dneighbourRanges, neighbourRangesTextureDesc, sizeof(int2)*graph.neighbourRanges.size());
	
	cudaChannelFormatDesc neighboursTextureDesc = cudaCreateChannelDesc<int>();

	neighboursTexture.addressMode[0] = cudaAddressModeWrap;
	neighboursTexture.filterMode = cudaFilterModePoint;
	neighboursTexture.normalized = false;
	cudaBindTexture(0, neighboursTexture, (void *)dneighbours, neighboursTextureDesc, sizeof(int)*graph.neighbours.size());

	//Allocate necessary buffers on the device.
	int *dmatch, *drequests;

	if (cudaMalloc(&dmatch, sizeof(int)*graph.nrVertices) != cudaSuccess
		|| cudaMalloc(&drequests, sizeof(int)*graph.nrVertices) != cudaSuccess)
	{
		cerr << "Not enough memory on device!" << endl;
		throw exception();
	}

	//Clear matching.
	if (cudaMemset(dmatch, 0, sizeof(int)*graph.nrVertices) != cudaSuccess)
	{
		cerr << "Unable to clear matching on device!" << endl;
		throw exception();
	}

	//Perform matching.
	int blocksPerGrid = (graph.nrVertices + threadsPerBlock - 1)/threadsPerBlock;
	
	//Perform all stages, one-by-one.
#ifndef NDEBUG
	cudaGetLastError();
#endif

	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);

#ifdef MATCH_INTERMEDIATE_COUNT
	cout << "0\t0\t0" << endl;
#endif

	for (int i = 0; i < NR_MATCH_ROUNDS; ++i)
	{
		gSelect<<<blocksPerGrid, threadsPerBlock>>>(dmatch, graph.nrVertices, rand());
		grRequest<<<blocksPerGrid, threadsPerBlock>>>(drequests, dmatch, graph.nrVertices);
		grRespond<<<blocksPerGrid, threadsPerBlock>>>(drequests, dmatch, graph.nrVertices);
		gMatch<<<blocksPerGrid, threadsPerBlock>>>(dmatch, drequests, graph.nrVertices);

#ifdef MATCH_INTERMEDIATE_COUNT
		cudaMemcpy(&match[0], dmatch, sizeof(int)*graph.nrVertices, cudaMemcpyDeviceToHost);
		
		double weight = 0;
		long size = 0;

		getWeight(weight, size, match, graph);

		cout << i + 1 << "\t" << weight << "\t" << size << endl;
#endif
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);

#ifndef NDEBUG
	cudaError_t error;

	if ((error = cudaGetLastError()) != cudaSuccess)
	{
		cerr << "A CUDA error occurred during the matching process: " << cudaGetErrorString(error) << endl;
		throw exception();
	}
#endif

	//Copy obtained matching on the device back to the host.
	if (cudaMemcpy(&match[0], dmatch, sizeof(int)*graph.nrVertices, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cerr << "Unable to retrieve data!" << endl;
		throw exception();
	}

	//Free memory.
	cudaFree(drequests);
	cudaFree(dmatch);
	cudaUnbindTexture(neighboursTexture);
	cudaUnbindTexture(neighbourRangesTexture);
}

void GraphMatchingGPURandomMaximal::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2) const
{
	//Creates a greedy random maximal matching on the GPU using atomic operations.
	//Assumes the current matching is empty.

	assert((int)match.size() == graph.nrVertices);
	
	//Setup textures.
	cudaChannelFormatDesc neighbourRangesTextureDesc = cudaCreateChannelDesc<int2>();

	neighbourRangesTexture.addressMode[0] = cudaAddressModeWrap;
	neighbourRangesTexture.filterMode = cudaFilterModePoint;
	neighbourRangesTexture.normalized = false;
	cudaBindTexture(0, neighbourRangesTexture, (void *)dneighbourRanges, neighbourRangesTextureDesc, sizeof(int2)*graph.neighbourRanges.size());
	
	cudaChannelFormatDesc neighboursTextureDesc = cudaCreateChannelDesc<int>();

	neighboursTexture.addressMode[0] = cudaAddressModeWrap;
	neighboursTexture.filterMode = cudaFilterModePoint;
	neighboursTexture.normalized = false;
	cudaBindTexture(0, neighboursTexture, (void *)dneighbours, neighboursTextureDesc, sizeof(int)*graph.neighbours.size());

	//Allocate necessary buffers on the device.
	int *dmatch, *drequests;

	if (cudaMalloc(&dmatch, sizeof(int)*graph.nrVertices) != cudaSuccess
		|| cudaMalloc(&drequests, sizeof(int)*graph.nrVertices) != cudaSuccess)
	{
		cerr << "Not enough memory on device!" << endl;
		throw exception();
	}

	//Clear matching.
	if (cudaMemset(dmatch, 0, sizeof(int)*graph.nrVertices) != cudaSuccess)
	{
		cerr << "Unable to clear matching on device!" << endl;
		throw exception();
	}

	//Perform matching.
	int keepMatching = 1, count = 0;
	int blocksPerGrid = (graph.nrVertices + threadsPerBlock - 1)/threadsPerBlock;

	//Perform all stages, one-by-one.
#ifndef NDEBUG
	cudaGetLastError();
#endif

	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);

	while (keepMatching == 1 && ++count < NR_MAX_MATCH_ROUNDS)
	{
		keepMatching = 0;
		cudaMemcpyToSymbol(dkeepMatching, &keepMatching, sizeof(int));

		gaSelect<<<blocksPerGrid, threadsPerBlock>>>(dmatch, graph.nrVertices, rand());
		grRequest<<<blocksPerGrid, threadsPerBlock>>>(drequests, dmatch, graph.nrVertices);
		grRespond<<<blocksPerGrid, threadsPerBlock>>>(drequests, dmatch, graph.nrVertices);
		gMatch<<<blocksPerGrid, threadsPerBlock>>>(dmatch, drequests, graph.nrVertices);

		cudaMemcpyFromSymbol(&keepMatching, dkeepMatching, sizeof(int));
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);

#ifndef NDEBUG
	cudaError_t error;

	if ((error = cudaGetLastError()) != cudaSuccess)
	{
		cerr << "A CUDA error occurred during the matching process: " << cudaGetErrorString(error) << endl;
		throw exception();
	}
#endif

	//Copy obtained matching on the device back to the host.
	if (cudaMemcpy(&match[0], dmatch, sizeof(int)*graph.nrVertices, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cerr << "Unable to retrieve data!" << endl;
		throw exception();
	}

	//Free memory.
	cudaFree(drequests);
	cudaFree(dmatch);
	cudaUnbindTexture(neighboursTexture);
	cudaUnbindTexture(neighbourRangesTexture);
}

void GraphMatchingGPUWeighted::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2) const
{
	//Creates a greedy weighted matching on the GPU.
	//Assumes the current matching is empty.

	assert((int)match.size() == graph.nrVertices);
	
	//Setup textures.
	cudaChannelFormatDesc neighbourRangesTextureDesc = cudaCreateChannelDesc<int2>();

	neighbourRangesTexture.addressMode[0] = cudaAddressModeWrap;
	neighbourRangesTexture.filterMode = cudaFilterModePoint;
	neighbourRangesTexture.normalized = false;
	cudaBindTexture(0, neighbourRangesTexture, (void *)dneighbourRanges, neighbourRangesTextureDesc, sizeof(int2)*graph.neighbourRanges.size());
	
	cudaChannelFormatDesc neighboursTextureDesc = cudaCreateChannelDesc<int>();

	neighboursTexture.addressMode[0] = cudaAddressModeWrap;
	neighboursTexture.filterMode = cudaFilterModePoint;
	neighboursTexture.normalized = false;
	cudaBindTexture(0, neighboursTexture, (void *)dneighbours, neighboursTextureDesc, sizeof(int)*graph.neighbours.size());

	cudaChannelFormatDesc weightsTextureDesc = cudaCreateChannelDesc<float>();

	weightsTexture.addressMode[0] = cudaAddressModeWrap;
	weightsTexture.filterMode = cudaFilterModePoint;
	weightsTexture.normalized = false;
	cudaBindTexture(0, weightsTexture, (void *)dweights, weightsTextureDesc, sizeof(float)*graph.neighbourWeights.size());

	//Allocate necessary buffers on the device.
	int *dmatch, *drequests;

	if (cudaMalloc(&dmatch, sizeof(int)*graph.nrVertices) != cudaSuccess
		|| cudaMalloc(&drequests, sizeof(int)*graph.nrVertices) != cudaSuccess)
	{
		cerr << "Not enough memory on device!" << endl;
		throw exception();
	}

	//Clear matching.
	if (cudaMemset(dmatch, 0, sizeof(int)*graph.nrVertices) != cudaSuccess)
	{
		cerr << "Unable to clear matching on device!" << endl;
		throw exception();
	}

	//Perform matching.
	int blocksPerGrid = (graph.nrVertices + threadsPerBlock - 1)/threadsPerBlock;

	//Perform all stages, one-by-one.
#ifndef NDEBUG
	cudaGetLastError();
#endif

	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);

#ifdef MATCH_INTERMEDIATE_COUNT
	cout << "0\t0\t0" << endl;
#endif

	for (int i = 0; i < NR_MATCH_ROUNDS; ++i)
	{
		gSelect<<<blocksPerGrid, threadsPerBlock>>>(dmatch, graph.nrVertices, rand());
		gwRequest<<<blocksPerGrid, threadsPerBlock>>>(drequests, dmatch, graph.nrVertices);
		gwRespond<<<blocksPerGrid, threadsPerBlock>>>(drequests, dmatch, graph.nrVertices);
		gMatch<<<blocksPerGrid, threadsPerBlock>>>(dmatch, drequests, graph.nrVertices);

#ifdef MATCH_INTERMEDIATE_COUNT
		cudaMemcpy(&match[0], dmatch, sizeof(int)*graph.nrVertices, cudaMemcpyDeviceToHost);
		
		double weight = 0;
		long size = 0;

		getWeight(weight, size, match, graph);

		cout << i + 1 << "\t" << weight << "\t" << size << endl;
#endif
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);

#ifndef NDEBUG
	cudaError_t error;

	if ((error = cudaGetLastError()) != cudaSuccess)
	{
		cerr << "A CUDA error occurred during the matching process: " << cudaGetErrorString(error) << endl;
		throw exception();
	}
#endif

	//Copy obtained matching on the device back to the host.
	if (cudaMemcpy(&match[0], dmatch, sizeof(int)*graph.nrVertices, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cerr << "Unable to retrieve data!" << endl;
		throw exception();
	}

	//Free memory.
	cudaFree(drequests);
	cudaFree(dmatch);

	cudaUnbindTexture(weightsTexture);
	cudaUnbindTexture(neighboursTexture);
	cudaUnbindTexture(neighbourRangesTexture);
}

void GraphMatchingGPUWeightedMaximal::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2) const
{
	//Creates a greedy weighted matching on the GPU.
	//Assumes the current matching is empty.

	assert((int)match.size() == graph.nrVertices);
	
	//Setup textures.
	cudaChannelFormatDesc neighbourRangesTextureDesc = cudaCreateChannelDesc<int2>();

	neighbourRangesTexture.addressMode[0] = cudaAddressModeWrap;
	neighbourRangesTexture.filterMode = cudaFilterModePoint;
	neighbourRangesTexture.normalized = false;
	cudaBindTexture(0, neighbourRangesTexture, (void *)dneighbourRanges, neighbourRangesTextureDesc, sizeof(int2)*graph.neighbourRanges.size());
	
	cudaChannelFormatDesc neighboursTextureDesc = cudaCreateChannelDesc<int>();

	neighboursTexture.addressMode[0] = cudaAddressModeWrap;
	neighboursTexture.filterMode = cudaFilterModePoint;
	neighboursTexture.normalized = false;
	cudaBindTexture(0, neighboursTexture, (void *)dneighbours, neighboursTextureDesc, sizeof(int)*graph.neighbours.size());

	cudaChannelFormatDesc weightsTextureDesc = cudaCreateChannelDesc<float>();

	weightsTexture.addressMode[0] = cudaAddressModeWrap;
	weightsTexture.filterMode = cudaFilterModePoint;
	weightsTexture.normalized = false;
	cudaBindTexture(0, weightsTexture, (void *)dweights, weightsTextureDesc, sizeof(float)*graph.neighbourWeights.size());

	//Allocate necessary buffers on the device.
	int *dmatch, *drequests;

	if (cudaMalloc(&dmatch, sizeof(int)*graph.nrVertices) != cudaSuccess
		|| cudaMalloc(&drequests, sizeof(int)*graph.nrVertices) != cudaSuccess)
	{
		cerr << "Not enough memory on device!" << endl;
		throw exception();
	}

	//Clear matching.
	if (cudaMemset(dmatch, 0, sizeof(int)*graph.nrVertices) != cudaSuccess)
	{
		cerr << "Unable to clear matching on device!" << endl;
		throw exception();
	}

	//Perform matching.
	int keepMatching = 1, count = 0;
	int blocksPerGrid = (graph.nrVertices + threadsPerBlock - 1)/threadsPerBlock;

	//Perform all stages, one-by-one.
#ifndef NDEBUG
	cudaGetLastError();
#endif

	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);

	while (keepMatching == 1 && ++count < NR_MAX_MATCH_ROUNDS)
	{
		keepMatching = 0;
		cudaMemcpyToSymbol(dkeepMatching, &keepMatching, sizeof(int));

		gaSelect<<<blocksPerGrid, threadsPerBlock>>>(dmatch, graph.nrVertices, rand());
		gwRequest<<<blocksPerGrid, threadsPerBlock>>>(drequests, dmatch, graph.nrVertices);
		gwRespond<<<blocksPerGrid, threadsPerBlock>>>(drequests, dmatch, graph.nrVertices);
		gMatch<<<blocksPerGrid, threadsPerBlock>>>(dmatch, drequests, graph.nrVertices);

		cudaMemcpyFromSymbol(&keepMatching, dkeepMatching, sizeof(int));
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);

#ifndef NDEBUG
	cudaError_t error;

	if ((error = cudaGetLastError()) != cudaSuccess)
	{
		cerr << "A CUDA error occurred during the matching process: " << cudaGetErrorString(error) << endl;
		throw exception();
	}
#endif

	//Copy obtained matching on the device back to the host.
	if (cudaMemcpy(&match[0], dmatch, sizeof(int)*graph.nrVertices, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cerr << "Unable to retrieve data!" << endl;
		throw exception();
	}

	//Free memory.
	cudaFree(drequests);
	cudaFree(dmatch);

	cudaUnbindTexture(weightsTexture);
	cudaUnbindTexture(neighboursTexture);
	cudaUnbindTexture(neighbourRangesTexture);
}

