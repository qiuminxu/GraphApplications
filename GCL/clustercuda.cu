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
#include <cstdlib>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_functions.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/scatter.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/adjacent_difference.h>

//CUDA 3.2 does not seem to make definitions for texture types.
#ifndef cudaTextureType1D
#define cudaTextureType1D 0x01
#endif

#include "clustercuda.h"

using namespace clu;
using namespace std;
using namespace thrust;

//We stop trying to increase the modularity if we drop below 10 percent of the modularity of the best encountered clustering.
#define MATCH_ITERATIONS 64
#define LEFTROTATE(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

//==== Kernel variables ====
__device__ int d_keepMatching;

texture<int2, cudaTextureType1D, cudaReadModeElementType> neighbourRangesTexture;
texture<int2, cudaTextureType1D, cudaReadModeElementType> neighboursTexture;
texture<int, cudaTextureType1D, cudaReadModeElementType> vertexWeightsTexture;
#ifndef LEAN
texture<float2, cudaTextureType1D, cudaReadModeElementType> coordinatesTexture;
#endif

//==== Matching kernels ====
/*
   Match values match[i] have the following interpretation for a vertex i:
   0   = blue,
   1   = red,
   2   = unmatchable (all neighbours of i have been matched),
   3   = reserved,
   >=4 = matched.
*/

//Determine blue and red groups using TEA-4.
__global__ void d_matchColour(int * const match, const int nrVertices, const uint random)
{
 	//See Figure 2 of 'GPU Random Numbers via the Tiny Encryption Algorithm', Zafar (2010).
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;

	//Can this vertex still be matched?
	if (match[i] >= 2) return;
	
	//Indicate that this matching is not yet maximal.
	d_keepMatching = 1;

	//Start hashing.
	uint sum = 0, v0 = i, v1 = random;
	
	sum += 0x9e3779b9;
	v0 += ((v1 << 4) + 0xa341316c)^(v1 + sum)^((v1 >> 5) + 0xc8013ea4);
	v1 += ((v0 << 4) + 0xad90777d)^(v0 + sum)^((v0 >> 5) + 0x7e95761e);
	
	sum += 0x9e3779b9;
	v0 += ((v1 << 4) + 0xa341316c)^(v1 + sum)^((v1 >> 5) + 0xc8013ea4);
	v1 += ((v0 << 4) + 0xad90777d)^(v0 + sum)^((v0 >> 5) + 0x7e95761e);
	
	sum += 0x9e3779b9;
	v0 += ((v1 << 4) + 0xa341316c)^(v1 + sum)^((v1 >> 5) + 0xc8013ea4);
	v1 += ((v0 << 4) + 0xad90777d)^(v0 + sum)^((v0 >> 5) + 0x7e95761e);
	
	sum += 0x9e3779b9;
	v0 += ((v1 << 4) + 0xa341316c)^(v1 + sum)^((v1 >> 5) + 0xc8013ea4);
	v1 += ((v0 << 4) + 0xad90777d)^(v0 + sum)^((v0 >> 5) + 0x7e95761e);
	
	match[i] = ((v0 + v1) < MATCH_BARRIER ? 0 : 1);
}

//Let all blue vertices propose to a red neighbour.
__global__ void d_matchPropose(int * const requests, const int * const match, const int nrVertices, const long twoOmega)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);
	const long wgt = tex1Dfetch(vertexWeightsTexture, i);

	//Look at all blue vertices and let them make requests.
	if (match[i] == 0)
	{
		long maxWeight = LONG_MIN;
		int candidate = nrVertices;
		int dead = 1;

		for (int j = indices.x; j < indices.y; ++j)
		{
			//Only propose to red neighbours.
			const int2 ni = tex1Dfetch(neighboursTexture, j);
			const int nm = match[ni.x];

			//Do we have an unmatched neighbour?
			if (nm < 4)
			{
				//Is this neighbour red?
				if (nm == 1)
				{
					//Propose to the heaviest neighbour.
					const long weight = twoOmega*(long)ni.y - wgt*(long)tex1Dfetch(vertexWeightsTexture, ni.x);

					if (weight > maxWeight)
					{
						maxWeight = weight;
						candidate = ni.x;
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

//Let each red vertex respond to one proposal.
__global__ void d_matchRespond(int * const requests, const int * const match, const int nrVertices, const long twoOmega)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);
	const long wgt = tex1Dfetch(vertexWeightsTexture, i);

	//Look at all red vertices.
	if (match[i] == 1)
	{
		long maxWeight = LONG_MIN;
		int candidate = nrVertices;

		//Select heaviest available proposer.
		for (int j = indices.x; j < indices.y; ++j)
		{
			const int2 ni = tex1Dfetch(neighboursTexture, j);

			//Only respond to blue neighbours.
			if (match[ni.x] == 0)
			{
				const long weight = twoOmega*(long)ni.y - wgt*(long)tex1Dfetch(vertexWeightsTexture, ni.x);

				if (weight > maxWeight)
				{
					if (requests[ni.x] == i)
					{
						maxWeight = weight;
						candidate = ni.x;
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

//Match compatible requests and responses.
__global__ void d_matchMatch(int * const match, const int * const requests, const int nrVertices)
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

//Match satellite vertices to prevent star graphs from stopping the coarsening process.
__global__ void d_matchMarkSatellites(int * const requests, const int * const match, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);
	const long wgt = indices.y - indices.x;
	const int m = match[i];
	
	//Check if this is a satellite vertex (only unmatched vertices can be satellites).
	long neighbourDegrees = 0;
	
	if (m < 4)
	{
		for (int j = indices.x; j < indices.y; ++j)
		{
			const int2 ni = tex1Dfetch(neighboursTexture, j);
			const int2 nindices = tex1Dfetch(neighbourRangesTexture, ni.x);

			neighbourDegrees += nindices.y - nindices.x;
		}
	}
	
	//Mark satellites red and non-satellites blue, in the requests array.
	requests[i] = ((MATCH_SATELLITE*wgt*wgt <= neighbourDegrees && m < 4) ? 0 : 1);
}

//Match all satellites (red) to their heaviest non-satellite (blue) neighbour.
__global__ void d_matchMatchSatellites(int * const match, const int * const requests, const int nrVertices, const long twoOmega)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);
	const long wgt = tex1Dfetch(vertexWeightsTexture, i);
	
	//Non-satellites are skipped.
	if (requests[i] != 0) return;
	
	//Find heaviest neighbour.
	long maxWeight = LONG_MIN;
	int candidate = nrVertices;

	for (int j = indices.x; j < indices.y; ++j)
	{
		const int2 ni = tex1Dfetch(neighboursTexture, j);

		if (requests[ni.x] != 0)
		{
			//Match to the heaviest non-satellite neighbour.
			const long weight = twoOmega*(long)ni.y - wgt*(long)tex1Dfetch(vertexWeightsTexture, ni.x);

			if (weight > maxWeight)
			{
				maxWeight = weight;
				candidate = ni.x;
			}
		}
	}
	
	//If such a neighbour has been found, perform a matching.
	if (candidate < nrVertices) match[i] = match[candidate];
}

//==== Coarsening kernels ====

//Calculate new vertex weights.
__global__ void d_coarsenCreateVertexWeights(int * const weights, const int * const kappaInv, const int * const permute, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 ki = make_int2(kappaInv[i], kappaInv[i + 1]);
	int w = 0;
	
	for (int j = ki.x; j < ki.y; ++j)
	{
		w += tex1Dfetch(vertexWeightsTexture, permute[j]);
	}
	
	weights[i] = w;
}

//Calculate new vertex coordinates.
#ifndef LEAN
__global__ void d_coarsenCreateVertexCoordinates(float2 * const coordinates, const int * const kappaInv, const int * const permute, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 ki = make_int2(kappaInv[i], kappaInv[i + 1]);
	float2 c = make_float2(0.0f, 0.0f);
	
	for (int j = ki.x; j < ki.y; ++j)
	{
		const float2 dc = tex1Dfetch(coordinatesTexture, permute[j]);
		
		c.x += dc.x;
		c.y += dc.y;
	}
	
	const float inv = 1.0f/(float)(ki.y - ki.x);
	
	coordinates[i] = make_float2(c.x*inv, c.y*inv);
}
#endif

//Setup counting array for a parallel scan to create the offsets in the graph's neighbour list.
__global__ void d_coarsenCountNeighbours(int * const sigma, const int * const kappaInv, const int * const permute, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 ki = make_int2(kappaInv[i], kappaInv[i + 1]);
	int s = 0;
	
	for (int j = ki.x; j < ki.y; ++j)
	{
		const int2 r = tex1Dfetch(neighbourRangesTexture, permute[j]);
		
		s += r.y - r.x;
	}
	
	sigma[i] = s;
}

//Copy all neighbours to the coarse graph.
__global__ void d_coarsenCopyNeighbours(int2 * const neighbours, const int * const sigma, const int * const kappaInv, const int * const permute, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 ki = make_int2(kappaInv[i], kappaInv[i + 1]);
	int count = sigma[i];
	
	for (int j = ki.x; j < ki.y; ++j)
	{
		const int2 r = tex1Dfetch(neighbourRangesTexture, permute[j]);
		
		for (int k = r.x; k < r.y; ++k)
		{
			neighbours[count++] = tex1Dfetch(neighboursTexture, k);
		}
	}
}

//Apply projection map to all neighbours.
__global__ void d_coarsenProjectNeighbours(int2 * const neighbours, const int * const sigma, const int * const kappa, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 si = make_int2(sigma[i], sigma[i + 1]);
	
	for (int j = si.x; j < si.y; ++j)
	{
		const int2 n = neighbours[j];
		
		neighbours[j] = make_int2(kappa[n.x], n.y);
	}
}

//Perform heap-sort on the GPU.
__device__ void d_coarsenHeapSortSift(int2 * const list, int root, const int bottom)
{
	int max;
	
	while (root*2 <= bottom)
	{
		if (root*2 == bottom) max = root*2;
		else if (list[root*2].x > list[root*2 + 1].x) max = root*2;
		else max = root*2 + 1;
		
		if (list[root].x < list[max].x)
		{
			int2 tmp = list[root];
			
			list[root] = list[max];
			list[max] = tmp;
			root = max;
		}
		else
		{
			//Indicate that we are done.
			root = bottom + 1;
		}
	}
}

//Heap sort implementation from http://www.algorithmist.com/index.php/Heap_sort.c.
__device__ void d_coarsenHeapSort(int2 * const list, const int size)
{
	for (int i = size/2; i >= 0; --i)
	{
		d_coarsenHeapSortSift(list, i, size - 1);
	}
	
	for (int i = size - 1; i >= 1; --i)
	{
		int2 tmp = list[0];
		
		list[0] = list[i];
		list[i] = tmp;
		d_coarsenHeapSortSift(list, 0, i - 1);
	}
}

//Compress projected neighbour lists.
__global__ void d_coarsenCompressNeighbours(int2 * const neighbourRanges, int2 * const neighbours, const int * const sigma, const int nrVertices)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 si = make_int2(sigma[i], sigma[i + 1]);
	
	//Do nothing if the neighbour range is empty.
	if (si.x == si.y)
	{
		neighbourRanges[i] = make_int2(si.x, si.y);
		return;
	}
	
	//Sort neighbour ranges.
	d_coarsenHeapSort(&neighbours[si.x], si.y - si.x);
	
	//Extract unique neighbours.
	//Do NOT store self-edges, these are invalid for matching.
	int count = si.x;
	int2 n = neighbours[si.x];
	
	for (int j = si.x + 1; j < si.y; ++j)
	{
		const int2 n2 = neighbours[j];
		
		if (n2.x == n.x)
		{
			//Sum weights of joined edges.
			n.y += n2.y;
		}
		else
		{
			//We encounter a new neighbour, so store the current one.
			if (n.x != i) neighbours[count++] = n;
			
			n = n2;
		}
	}
	
	if (n.x != i) neighbours[count++] = n;
	
	neighbourRanges[i] = make_int2(si.x, count);
}

//==== Clustering kernels ====
__global__ void d_clusterGetModularity(long * const clusterMods, const int nrVertices, const long twoOmega)
{
	const int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i >= nrVertices) return;
	
	const int2 indices = tex1Dfetch(neighbourRangesTexture, i);
	const long wgt = tex1Dfetch(vertexWeightsTexture, i);
	long extWgt = 0;

	//Sum external edge weights.
	for (int j = indices.x; j < indices.y; ++j)
	{
		const int2 ni = tex1Dfetch(neighboursTexture, j);
		
		extWgt += (long)ni.y;
	}
	
	//Store contribution to modularity.
	clusterMods[i] = wgt*(twoOmega - wgt) - twoOmega*extWgt;
}

//==== CUDA kernels end ====

ClusterCUDA::ClusterCUDA(const int &_blockSize) :
	Cluster(),
	blockSize(_blockSize)
{
	if (blockSize <= 0)
	{
		cerr << "Invalid CUDA block size (" << blockSize << ")!" << endl;
		
		throw exception();
	}
}

ClusterCUDA::~ClusterCUDA()
{

}

void ClusterCUDA::match(int * const d_match, int * const d_requests, const int &h_nrVertices, const long &h_twoOmega) const
{
	//Keep matching until we obtain a maximal matching.
	int count = 0;
	int h_keepMatching = 1;
	const int nrBlocks = (h_nrVertices + blockSize - 1)/blockSize;
	
	//Clear matching values.
	cudaMemset(d_match, 0, h_nrVertices*sizeof(int));
	
	//Perform matching rounds.
	while (h_keepMatching != 0 && ++count < MATCH_ITERATIONS)
	{
		h_keepMatching = 0;
		cudaMemcpyToSymbol(d_keepMatching, &h_keepMatching, sizeof(int));
		
		d_matchColour<<<nrBlocks, blockSize>>>(d_match, h_nrVertices, rand());
		d_matchPropose<<<nrBlocks, blockSize>>>(d_requests, d_match, h_nrVertices, h_twoOmega);
		d_matchRespond<<<nrBlocks, blockSize>>>(d_requests, d_match, h_nrVertices, h_twoOmega);
		d_matchMatch<<<nrBlocks, blockSize>>>(d_match, d_requests, h_nrVertices);

		cudaMemcpyFromSymbol(&h_keepMatching, d_keepMatching, sizeof(int));
	}
	
	//Get rid of satellites.
	d_matchMarkSatellites<<<nrBlocks, blockSize>>>(d_requests, d_match, h_nrVertices);
	d_matchMatchSatellites<<<nrBlocks, blockSize>>>(d_match, d_requests, h_nrVertices, h_twoOmega);

#ifndef NDEBUG
	if (count >= MATCH_ITERATIONS)
	{
		cerr << "Warning: exceeded " << MATCH_ITERATIONS << " iterations during matching!" << endl;
	}
#endif
}

void ClusterCUDA::printDeviceVector(const device_vector<int> &v) const
{
	vector<int> w(v.size(), 0);
	
	thrust::copy(v.begin(), v.end(), w.begin());
	
	for (vector<int>::const_iterator i = w.begin(); i != w.end(); ++i) cerr << *i << " ";
	cerr << endl;
}

void ClusterCUDA::printDeviceVector(const device_vector<int2> &v) const
{
	vector<int2> w(v.size(), make_int2(0, 0));
	
	thrust::copy(v.begin(), v.end(), w.begin());
	
	for (vector<int2>::const_iterator i = w.begin(); i != w.end(); ++i) cerr << "(" << i->x << ", " << i->y << ") ";
	//for (vector<int2>::const_iterator i = w.begin(); i != w.end(); ++i) cerr << i->x << " ";
	cerr << endl;
}

Graph ClusterCUDA::createGraph(const device_vector<int2> &d_neighbourRanges,
				const device_vector<int2> &d_neighbours,
				const device_vector<int> &d_vertexWeights,
				const device_vector<float2> &d_coordinates,
				const int &h_nrVertices) const
{
	Graph graph;
	
	graph.nrVertices = h_nrVertices;
	//TODO: nrEdges.
	graph.neighbourRanges.resize(h_nrVertices);
	graph.neighbours.resize(d_neighbours.size());
	graph.vertexWeights.resize(h_nrVertices);
	graph.coordinates.resize(h_nrVertices);
	
	thrust::copy(d_neighbourRanges.begin(), d_neighbourRanges.begin() + h_nrVertices, graph.neighbourRanges.begin());
	thrust::copy(d_neighbours.begin(), d_neighbours.end(), graph.neighbours.begin());
	thrust::copy(d_vertexWeights.begin(), d_vertexWeights.begin() + h_nrVertices, graph.vertexWeights.begin());
	thrust::copy(d_coordinates.begin(), d_coordinates.begin() + h_nrVertices, graph.coordinates.begin());
	
	return graph;
}

//Count vertices that are either unmatched, or have a different matching value.
struct count_matched
{
	__host__ __device__ int operator () (const int &lhs, const int &rhs) const
	{
		return ((lhs < 4 || lhs != rhs) ? 1 : 0);
	};
};

struct is_nonzero
{
	__host__ __device__ bool operator () (const int &lhs) const
	{
		return (lhs != 0);
	};
};

#define REALMACHINE
vector<int> ClusterCUDA::cluster(const Graph &graph, const double &quality, Drawer *drawer) const
{
	//Draw the graph in its original state.
	if (drawer)
	{
		drawer->drawGraphMatrix(graph);
		drawer->drawGraphCoordinates(graph);
	}
	
	//==== Allocate memory.
	//Variables for matching.

#ifdef REALMACHINE
	device_vector<int> d_match(graph.nrVertices);
	device_vector<int> d_pi(graph.nrVertices);
	device_vector<long> d_deltas(graph.nrVertices);
	
	//Variables for coarsening.
	device_vector<int> d_sigma(graph.nrVertices);
	device_vector<int> d_kappa(graph.nrVertices);
	device_vector<int> d_kappaInv(graph.nrVertices);
	
	//Variables for clustering.
	vector<int> h_clustering(graph.nrVertices);
	device_vector<int> d_clustering(graph.nrVertices);
	device_vector<long> d_clusterMods(graph.nrVertices);

	//Two graphs we will alternate between.
	int h_nrVertices[2] = {0, 0};
	device_vector<int2> d_neighbourRanges[2] = {device_vector<int2>(graph.nrVertices), device_vector<int2>(graph.nrVertices)};
	device_vector<int2> d_neighbours[2] = {device_vector<int2>(graph.neighbours.size()), device_vector<int2>(graph.neighbours.size())};
	device_vector<int> d_vertexWeights[2] = {device_vector<int>(graph.nrVertices), device_vector<int>(graph.nrVertices)};
#else
	int* d_match;
	cudaMalloc(&d_match, graph.nrVertices * sizeof(int));

	//The pi array is used for proposals/responses as well as storing a permutation later to save memory.
	int* d_pi;
	cudaMalloc(&d_pi, graph.nrVertices * sizeof(int));
	long* d_deltas;
	cudaMalloc(&d_deltas, graph.nrVertices * sizeof(long));
	
	//Variables for coarsening.
	int* d_sigma;
	cudaMalloc(&d_sigma, graph.nrVertices * sizeof(int));
	int* d_kappa;
	cudaMalloc(&d_kappa, graph.nrVertices * sizeof(int));
	int* d_kappaInv;
	cudaMalloc(&d_kappaInv, graph.nrVertices * sizeof(int));

	//Variables for clustering.
	vector<int> h_clustering(graph.nrVertices);
	int* d_clustering;
	cudaMalloc(&d_clustering, graph.nrVertices * sizeof(int));
	long* d_clusterMods;
	cudaMalloc(&d_clusterMods, graph.nrVertices * sizeof(long));

	//Two graphs we will alternate between.
	int h_nrVertices[2] = {0, 0};
	int2* d_neighbourRanges;
	cudaMalloc(&d_neighbourRanges, graph.nrVertices * sizeof(int2) * 2);
	int2* d_neighbours;
	cudaMalloc(&d_neighbours, graph.neighbours.size() * sizeof(int2) * 2);
	int* d_vertexWeights;
	cudaMalloc(&d_vertexWeights, graph.nrVertices*sizeof(int));
#endif // ifdef REALMACHINE
	
#ifndef LEAN
	device_vector<float2> d_coordinates[2] = {device_vector<float2>(graph.nrVertices), device_vector<float2>(graph.nrVertices)};
#endif
	
	//==== Setup texture formats.
	cudaChannelFormatDesc neighbourRangesTextureDesc = cudaCreateChannelDesc<int2>();

	neighbourRangesTexture.addressMode[0] = cudaAddressModeWrap;
	neighbourRangesTexture.filterMode = cudaFilterModePoint;
	neighbourRangesTexture.normalized = false;
	
	cudaChannelFormatDesc neighboursTextureDesc = cudaCreateChannelDesc<int2>();

	neighboursTexture.addressMode[0] = cudaAddressModeWrap;
	neighboursTexture.filterMode = cudaFilterModePoint;
	neighboursTexture.normalized = false;
	
	cudaChannelFormatDesc vertexWeightsTextureDesc = cudaCreateChannelDesc<int>();

	vertexWeightsTexture.addressMode[0] = cudaAddressModeWrap;
	vertexWeightsTexture.filterMode = cudaFilterModePoint;
	vertexWeightsTexture.normalized = false;
	
#ifndef LEAN
	cudaChannelFormatDesc coordinatesTextureDesc = cudaCreateChannelDesc<float2>();

	coordinatesTexture.addressMode[0] = cudaAddressModeWrap;
	coordinatesTexture.filterMode = cudaFilterModePoint;
	coordinatesTexture.normalized = false;
#endif
	
	//==== Copy graph.
	const long h_twoOmega = 2L*graph.Omega;
	
#ifdef REALMACHINE
	h_nrVertices[0] = graph.nrVertices;
	thrust::copy(graph.neighbourRanges.begin(), graph.neighbourRanges.end(), d_neighbourRanges[0].begin());
	thrust::copy(graph.neighbours.begin(), graph.neighbours.end(), d_neighbours[0].begin());
	thrust::copy(graph.vertexWeights.begin(), graph.vertexWeights.end(), d_vertexWeights[0].begin());
#ifndef LEAN
	thrust::copy(graph.coordinates.begin(), graph.coordinates.end(), d_coordinates[0].begin());
#endif

#else
	h_nrVertices[0] = graph.nrVertices;
	cudaMemcpy(d_neighbourRanges, &graph.neighbourRanges.begin(), graph.neighbourRanges.size(), cudaMemcpyHostToDevice);
	cudaMemcpy(d_neighbours, &graph.neighbours.begin(), graph.neighbours.size(), cudaMemcpyHostToDevice);	
	cudaMemcpy(d_vertexWeights, &graph.vertexWeights.begin(), graph.vertexWeights.size(), cudaMemcpyHostToDevice);
#ifndef LEAN
	cudaMemcpy(d_coordinates, graph.coordinates, graph.coordinates.size(), cudaMemcpyHostToDevice);
#endif

#endif // ifdef REALMACHINE
	
	cerr << "cluster()-1" << graph.nrVertices<< endl;

	//==== Start coarsening.
	//Coarsen the graph recursively by merging clusters whose merge would increase modularity the most.
	
	//Assign initial clustering indices.
	thrust::sequence(d_clustering.begin(), d_clustering.end());
	
	//Start coarsening.
	bool done = false;
	int lastNrVertices = graph.nrVertices;
	long maxMod = LONG_MIN;
	int nrLevels = 0;
	int cur = 0;
	
#ifndef NDEBUG
	vector<int> h_tmpClustering(graph.nrVertices);
	vector<int> h_tmpPermute(graph.nrVertices);
	vector<int> h_tmpKappa(graph.nrVertices);
	vector<int> h_tmpKappaInv(graph.nrVertices);
#endif
	
	do
	{
		//Stop coarsening if the modularity cannot further be increased.
		lastNrVertices = h_nrVertices[cur];
		
		//Bind textures.
		cudaBindTexture(0, neighbourRangesTexture, raw_pointer_cast(&d_neighbourRanges[cur][0]), neighbourRangesTextureDesc, sizeof(int2)*d_neighbourRanges[cur].size());
		cudaBindTexture(0, neighboursTexture, raw_pointer_cast(&d_neighbours[cur][0]), neighboursTextureDesc, sizeof(int2)*d_neighbours[cur].size());
		cudaBindTexture(0, vertexWeightsTexture, raw_pointer_cast(&d_vertexWeights[cur][0]), vertexWeightsTextureDesc, sizeof(int2)*d_vertexWeights[cur].size());
#ifndef LEAN
		cudaBindTexture(0, coordinatesTexture, raw_pointer_cast(&d_coordinates[cur][0]), coordinatesTextureDesc, sizeof(int2)*d_coordinates[cur].size());
#endif
		
		//== Calculate modularity of current clustering and copy if it is the best so far.
		const int nrFineBlocks = (h_nrVertices[cur] + blockSize - 1)/blockSize;
		
		d_clusterGetModularity<<<nrFineBlocks, blockSize>>>(raw_pointer_cast(&d_clusterMods[0]), h_nrVertices[cur], h_twoOmega);
		
		const long mod = thrust::reduce(d_clusterMods.begin(), d_clusterMods.begin() + h_nrVertices[cur]);
		
		if (mod >= maxMod)
		{
			maxMod = mod;
			//TODO: Make this asynchronous.
			thrust::copy(d_clustering.begin(), d_clustering.end(), h_clustering.begin());
		}
		else if (mod < CLUSTER_STOPFRAC*maxMod)
		{
			//If we are moving away from the maximum, stop clustering.
			done = true;
		}
		
		//Check modularity.
#ifndef NDEBUG
		thrust::copy(d_clustering.begin(), d_clustering.end(), h_tmpClustering.begin());
		
		for (int i = 0; i < graph.nrVertices; ++i)
		{
			assert(h_tmpClustering[i] >= 0 && h_tmpClustering[i] < h_nrVertices[cur]);
		}
		
		cerr << (double)mod/(double)(4L*graph.Omega*graph.Omega) << " == " << modularity(graph, h_tmpClustering) << " (" << mod << ")" << endl;
		assert(fabs(((double)mod/(double)(4L*graph.Omega*graph.Omega)) - modularity(graph, h_tmpClustering)) < 1.0e-8);
#endif
		
		//== Generate matching and calculate change in modularity.
		match(raw_pointer_cast(&d_match[0]), raw_pointer_cast(&d_pi[0]), h_nrVertices[cur], h_twoOmega);

#ifndef NDEBUG
		//Verify matching.
		//With satellite matching, match() does not generate valid matchings.
		if (false)
		{
			std::vector<int> h_tmpMatch(h_nrVertices[cur]);
			Graph tmpGraph = createGraph(d_neighbourRanges[cur], d_neighbours[cur], d_vertexWeights[cur], d_coordinates[cur], h_nrVertices[cur]);
			
			thrust::copy(d_match.begin(), d_match.begin() + h_nrVertices[cur], h_tmpMatch.begin());
			assert(GraphMatching::test(h_tmpMatch, tmpGraph));
		}
#endif
		
		//== Coarsen graph.
		
		//Start with the identity permutation.
		thrust::sequence(d_pi.begin(), d_pi.begin() + h_nrVertices[cur]);
		
		//Sort by matching array.
		thrust::sort_by_key(d_match.begin(), d_match.begin() + h_nrVertices[cur], d_pi.begin());
		
		//Re-enumerate matching array.
		thrust::adjacent_difference(d_match.begin(), d_match.begin() + h_nrVertices[cur], d_sigma.begin(), count_matched());
		
		//Create inverse mapping.
		d_sigma[0] = 1;
		h_nrVertices[1 - cur] = thrust::copy_if(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(0) + h_nrVertices[cur], d_sigma.begin(), d_kappaInv.begin(), is_nonzero()) - d_kappaInv.begin();
		//FIXME: thrust::copy_if ensures that the results are ordered, otherwise we will need an additional sort.
		//thrust::sort(d_kappaInv.begin(), d_kappaInv.begin() + h_nrVertices[1 - cur]);
		d_kappaInv[h_nrVertices[1 - cur]] = h_nrVertices[cur];
		
		//TODO: Try in-place scan.
		d_sigma[0] = 0;
		thrust::inclusive_scan(d_sigma.begin(), d_sigma.begin() + h_nrVertices[cur], d_match.begin());
		
#ifndef NDEBUG	
		cerr << "Coarsened to " << h_nrVertices[1 - cur] << "/" << h_nrVertices[cur] << " (" << (100*h_nrVertices[1 - cur])/h_nrVertices[cur] << "%) vertices." << endl;
		//Now that we match satellites, it can be true that the graph is coarsened more agressively.
		//assert(2*h_nrVertices[1 - cur] >= h_nrVertices[cur]);
#endif
		//Distribute matching array according to the permutation.
		thrust::scatter(d_match.begin(), d_match.begin() + h_nrVertices[cur], d_pi.begin(), d_kappa.begin());
		
		//Apply coarsening.
		thrust::gather(d_clustering.begin(), d_clustering.end(), d_kappa.begin(), d_clustering.begin());
		
#ifndef NDEBUG
		//Verify that the projection and inverse maps are properly generated in debug mode.
		thrust::copy(d_pi.begin(), d_pi.end(), h_tmpPermute.begin());
		thrust::copy(d_kappa.begin(), d_kappa.end(), h_tmpKappa.begin());
		thrust::copy(d_kappaInv.begin(), d_kappaInv.end(), h_tmpKappaInv.begin());
		
		//Check permutation.
		if (true)
		{
			vector<bool> check(h_nrVertices[cur], false);
			
			for (int i = 0; i < h_nrVertices[cur]; ++i)
			{
				const int j = h_tmpPermute[i];
				
				assert(j >= 0 && j < h_nrVertices[cur]);
				check[j] = true;
			}
			
			for (int i = 0; i < h_nrVertices[cur]; ++i)
			{
				assert(check[i]);
			}
		}
		
		for (int i = 0; i < h_nrVertices[cur]; ++i)
		{
			assert(h_tmpKappa[i] >= 0 && h_tmpKappa[i] < h_nrVertices[1 - cur]);
		}
		
		for (int i = 0; i < h_nrVertices[1 - cur]; ++i)
		{
			assert(h_tmpKappaInv[i] < h_tmpKappaInv[i + 1]);
			assert(h_tmpKappaInv[i] >= 0 && h_tmpKappaInv[i] < h_nrVertices[cur]);
			
			const int k = h_tmpKappa[h_tmpPermute[h_tmpKappaInv[i]]];
			
			for (int j = h_tmpKappaInv[i]; j < h_tmpKappaInv[i + 1]; ++j)
			{
				assert(h_tmpKappa[h_tmpPermute[j]] == k);
			}
		}
#endif
		
		//Start creating the coarse graph.
		const int nrCoarseBlocks = (h_nrVertices[1 - cur] + blockSize - 1)/blockSize;
		
		//Calculate vertex weights.
		d_coarsenCreateVertexWeights<<<nrCoarseBlocks, blockSize>>>(raw_pointer_cast(&d_vertexWeights[1 - cur][0]), raw_pointer_cast(&d_kappaInv[0]), raw_pointer_cast(&d_pi[0]), h_nrVertices[1 - cur]);
#ifndef LEAN
		d_coarsenCreateVertexCoordinates<<<nrCoarseBlocks, blockSize>>>(raw_pointer_cast(&d_coordinates[1 - cur][0]), raw_pointer_cast(&d_kappaInv[0]), raw_pointer_cast(&d_pi[0]), h_nrVertices[1 - cur]);
#endif
		
		//Create new neighbour lists for the coarse graph.
		d_coarsenCountNeighbours<<<nrCoarseBlocks, blockSize>>>(raw_pointer_cast(&d_match[0]), raw_pointer_cast(&d_kappaInv[0]), raw_pointer_cast(&d_pi[0]), h_nrVertices[1 - cur]);
		d_match[h_nrVertices[1 - cur]] = 0;
		thrust::exclusive_scan(d_match.begin(), d_match.begin() + h_nrVertices[1 - cur] + 1, d_sigma.begin());
		d_coarsenCopyNeighbours<<<nrCoarseBlocks, blockSize>>>(raw_pointer_cast(&d_neighbours[1 - cur][0]), raw_pointer_cast(&d_sigma[0]), raw_pointer_cast(&d_kappaInv[0]), raw_pointer_cast(&d_pi[0]), h_nrVertices[1 - cur]);
		d_coarsenProjectNeighbours<<<nrCoarseBlocks, blockSize>>>(raw_pointer_cast(&d_neighbours[1 - cur][0]), raw_pointer_cast(&d_sigma[0]), raw_pointer_cast(&d_kappa[0]), h_nrVertices[1 - cur]);
		d_coarsenCompressNeighbours<<<nrCoarseBlocks, blockSize>>>(raw_pointer_cast(&d_neighbourRanges[1 - cur][0]), raw_pointer_cast(&d_neighbours[1 - cur][0]), raw_pointer_cast(&d_sigma[0]), h_nrVertices[1 - cur]);
		
		//Unbind textures.
#ifndef LEAN
		cudaUnbindTexture(coordinatesTexture);
#endif
		cudaUnbindTexture(vertexWeightsTexture);
		cudaUnbindTexture(neighboursTexture);
		cudaUnbindTexture(neighbourRangesTexture);
		
		//Descend.
		cur = 1 - cur;
		++nrLevels;
		
		//Draw intermediate clustering.
#ifndef LEAN
		if (drawer)
		{
			Graph tmpGraph = createGraph(d_neighbourRanges[cur], d_neighbours[cur], d_vertexWeights[cur], d_coordinates[cur], h_nrVertices[cur]);
			
			drawer->drawGraphMatrix(tmpGraph);
			drawer->drawGraphCoordinates(tmpGraph);
		}
#endif
	}
	while (h_nrVertices[cur] < lastNrVertices && !done);
	
	//Draw the graph final clustering.
	if (drawer)
	{
		drawer->drawGraphMatrix(graph);
		drawer->drawGraphClustering(graph, h_clustering);
	}
	
#ifndef NDEBUG
	cerr << "Created a clustering of " << graph.nrVertices << " vertices into " << lastNrVertices << " components in " << nrLevels << " iterations (CUDA)." << endl;
#endif
	
	return h_clustering;
}

