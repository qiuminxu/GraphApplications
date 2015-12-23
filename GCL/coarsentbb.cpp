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
#include <set>
#include <algorithm>

#include <tbb/tbb.h>

#include "tbbext.h"
#include "match.h"
#include "graph.h"
#include "coarsentbb.h"

using namespace std;
using namespace tbb;
using namespace clu;

GraphCoarseningTBB::GraphCoarseningTBB()
{

}

GraphCoarseningTBB::~GraphCoarseningTBB()
{

}

struct count_matched
{
	int operator () (const int &lhs, const int &rhs) const
	{
		return ((!MATCH_MATCHED(lhs) || lhs != rhs) ? 1 : 0);
	};
};

struct is_nonzero
{
	bool operator () (const int &lhs) const
	{
		return (lhs != 0);
	};
};

class SumVertexWeights
{
	public:
		SumVertexWeights(const vector<int>::iterator &_weights, const vector<int>::const_iterator _kappaInv, const vector<int>::const_iterator &_permute, const vector<int>::const_iterator &_fineWeights) :
			weights(_weights), kappaInv(_kappaInv), permute(_permute), fineWeights(_fineWeights)
		{};
		
		~SumVertexWeights() {};
		
		void operator () (const blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				const int2 ki = make_int2(kappaInv[i], kappaInv[i + 1]);
				int wgt = 0;
				
				for (int j = ki.x; j < ki.y; ++j) wgt += fineWeights[permute[j]];
				
				weights[i] = wgt;
			}
		};
		
	private:
		vector<int>::iterator weights;
		vector<int>::const_iterator kappaInv;
		vector<int>::const_iterator permute;
		vector<int>::const_iterator fineWeights;
};

#ifndef LEAN
class SumVertexCoordinates
{
	public:
		SumVertexCoordinates(const vector<float2>::iterator &_coords, const vector<int>::const_iterator _kappaInv, const vector<int>::const_iterator &_permute, const vector<float2>::const_iterator &_fineCoords) :
			coords(_coords), kappaInv(_kappaInv), permute(_permute), fineCoords(_fineCoords)
		{};
		
		~SumVertexCoordinates() {};
		
		void operator () (const blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				const int2 ki = make_int2(kappaInv[i], kappaInv[i + 1]);
				float2 coord = make_float2(0.0f, 0.0f);
				
				for (int j = ki.x; j < ki.y; ++j)
				{
					const float2 dc = fineCoords[permute[j]];
					coord.x += dc.x; coord.y += dc.y;
				}
				
				coords[i] = make_float2(coord.x/(float)(ki.y - ki.x), coord.y/(float)(ki.y - ki.x));
			}
		};
		
	private:
		vector<float2>::iterator coords;
		vector<int>::const_iterator kappaInv;
		vector<int>::const_iterator permute;
		vector<float2>::const_iterator fineCoords;
};
#endif

class SumNeighbourRanges
{
	public:
		SumNeighbourRanges(const vector<int>::iterator &_sigma, const vector<int>::const_iterator _kappaInv, const vector<int>::const_iterator &_permute, const vector<int2>::const_iterator &_fineRanges) :
			sigma(_sigma), kappaInv(_kappaInv), permute(_permute), fineRanges(_fineRanges)
		{};
		
		~SumNeighbourRanges() {};
		
		void operator () (const blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				const int2 ki = make_int2(kappaInv[i], kappaInv[i + 1]);
				int count = 0;
				
				for (int j = ki.x; j < ki.y; ++j)
				{
					const int2 ni = fineRanges[permute[j]];
					
					count += ni.y - ni.x;
				}
				
				sigma[i] = count;
			}
		};
		
	private:
		vector<int>::iterator sigma;
		vector<int>::const_iterator kappaInv;
		vector<int>::const_iterator permute;
		vector<int2>::const_iterator fineRanges;
};

class CopyNeighbours
{
	public:
		CopyNeighbours(const vector<int2>::iterator &_neighbours, const vector<int>::const_iterator &_sigma, const vector<int>::const_iterator _kappaInv, const vector<int>::const_iterator &_permute, const vector<int2>::const_iterator &_fineRanges, const vector<int2>::const_iterator &_fineNeighbours) :
			neighbours(_neighbours), sigma(_sigma), kappaInv(_kappaInv), permute(_permute), fineRanges(_fineRanges), fineNeighbours(_fineNeighbours)
		{};
		
		~CopyNeighbours() {};
		
		void operator () (const blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				const int2 ki = make_int2(kappaInv[i], kappaInv[i + 1]);
				int count = sigma[i];
				
				for (int j = ki.x; j < ki.y; ++j)
				{
					const int2 ni = fineRanges[permute[j]];
					
					for (int k = ni.x; k < ni.y; ++k) neighbours[count++] = fineNeighbours[k];
				}
			}
		};
		
	private:
		vector<int2>::iterator neighbours;
		vector<int>::const_iterator sigma;
		vector<int>::const_iterator kappaInv;
		vector<int>::const_iterator permute;
		vector<int2>::const_iterator fineRanges;
		vector<int2>::const_iterator fineNeighbours;
};

class ProjectNeighbours
{
	public:
		ProjectNeighbours(const vector<int2>::iterator &_neighbours, const vector<int>::const_iterator &_sigma, const vector<int>::const_iterator _kappa) :
			neighbours(_neighbours), sigma(_sigma), kappa(_kappa)
		{};
		
		~ProjectNeighbours() {};
		
		void operator () (const blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				const int2 si = make_int2(sigma[i], sigma[i + 1]);
				
				for (int j = si.x; j < si.y; ++j)
				{
					const int2 ni = neighbours[j];
					
					neighbours[j] = make_int2(kappa[ni.x], ni.y);
				}
			}
		};
		
	private:
		vector<int2>::iterator neighbours;
		vector<int>::const_iterator sigma;
		vector<int>::const_iterator kappa;
};

class CompressNeighbours
{
	public:
		CompressNeighbours(const vector<int2>::iterator &_ranges, const vector<int2>::iterator &_neighbours, const vector<int>::const_iterator &_sigma) :
			ranges(_ranges), neighbours(_neighbours), sigma(_sigma)
		{};
		
		~CompressNeighbours() {};
		
		void operator () (const blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				const int2 si = make_int2(sigma[i], sigma[i + 1]);
				
				if (si.x == si.y)
				{
					ranges[i] = make_int2(si.x, si.x);
				}
				else
				{
					heapSort(&neighbours[si.x], si.y - si.x);
					
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
							if (n.x != (int)i) neighbours[count++] = n;
							
							n = n2;
						}
					}
					
					if (n.x != (int)i) neighbours[count++] = n;
					
					ranges[i] = make_int2(si.x, count);
				}
			}
		};
		
	private:
		vector<int2>::iterator ranges;
		vector<int2>::iterator neighbours;
		vector<int>::const_iterator sigma;

		//Heap sort implementation from http://www.algorithmist.com/index.php/Heap_sort.c.
		void heapSort(int2 * const list, const int &size) const
		{
			for (int i = size/2; i >= 0; --i)
			{
				heapSortSift(list, i, size - 1);
			}
			
			for (int i = size - 1; i >= 1; --i)
			{
				swap(list[0], list[i]);
				heapSortSift(list, 0, i - 1);
			}
		};
		
		void heapSortSift(int2 * const list, int root, const int &bottom) const
		{
			bool done = false;
			int max;
			
			while ((root*2 <= bottom) && !done)
			{
				if (root*2 == bottom) max = root*2;
				else if (list[root*2].x > list[root*2 + 1].x) max = root*2;
				else max = root*2 + 1;
				
				if (list[root].x < list[max].x)
				{
					swap(list[root], list[max]);
					root = max;
				}
				else
				{
					done = true;
				}
			}
		};
};

CoarsenedGraph GraphCoarseningTBB::coarsen(const Graph &fine, const vector<int> &_mu) const
{
	vector<int> mu = _mu;
	vector<int> sigma(fine.nrVertices);
	
	CoarsenedGraph c;
	
	//For going up the multilevel hierarchy, permute and kappaInv should be saved.
	vector<int> permute(fine.nrVertices);
	vector<int> kappaInv(fine.nrVertices);
	
	c.kappa.resize(fine.nrVertices);
	
	parallel_sequence(permute.begin(), permute.end());
	parallel_sort_by_key(mu.begin(), mu.end(), permute.begin());
	parallel_adjacent_difference(mu.begin(), mu.end(), sigma.begin(), count_matched());
	
	//Create inverse.
	sigma[0] = 1;
	c.graph.nrVertices = parallel_copy_if(sigma.begin(), sigma.end(), kappaInv.begin(), is_nonzero()) - kappaInv.begin();
	kappaInv.resize(c.graph.nrVertices + 1);
	kappaInv[c.graph.nrVertices] = fine.nrVertices;
	
	//Renumber matching values.
	sigma[0] = 0;
	parallel_inclusive_scan(sigma.begin(), sigma.end(), mu.begin());
	
	//And redistribute them to obtain kappa.
	parallel_scatter(mu.begin(), mu.end(), permute.begin(), c.kappa.begin());
	
	//Allocate memory for coarse graph.
	c.graph.neighbourRanges.resize(c.graph.nrVertices);
	c.graph.vertexWeights.resize(c.graph.nrVertices);
	c.graph.Omega = fine.Omega;
#ifndef LEAN
	c.graph.coordinates.resize(c.graph.nrVertices);
#endif

#ifndef NDEBUG	
	cerr << "Coarsened to " << c.graph.nrVertices << "/" << fine.nrVertices << " (" << (100*c.graph.nrVertices)/fine.nrVertices << "%) vertices." << endl;
		
	//Check permutation.
	if (true)
	{
		vector<bool> check(fine.nrVertices, false);
		
		for (int i = 0; i < fine.nrVertices; ++i)
		{
			const int j = permute[i];
			
			assert(j >= 0 && j < fine.nrVertices);
			check[j] = true;
		}
		
		for (int i = 0; i < fine.nrVertices; ++i)
		{
			assert(check[i]);
		}
	}
	
	//Check kappa and its inverse.
	for (int i = 0; i < fine.nrVertices; ++i)
	{
		assert(c.kappa[i] >= 0 && c.kappa[i] < c.graph.nrVertices);
	}
	
	for (int i = 0; i < c.graph.nrVertices; ++i)
	{
		assert(kappaInv[i] < kappaInv[i + 1]);
		assert(kappaInv[i] >= 0 && kappaInv[i] < fine.nrVertices);
		
		const int k = c.kappa[permute[kappaInv[i]]];
		
		for (int j = kappaInv[i]; j < kappaInv[i + 1]; ++j)
		{
			assert(c.kappa[permute[j]] == k);
		}
	}
#endif
	
	//Sum vertex weights and average vertex coordinates.
	parallel_for(blocked_range<size_t>(0, c.graph.nrVertices), SumVertexWeights(c.graph.vertexWeights.begin(), kappaInv.begin(), permute.begin(), fine.vertexWeights.begin()));
#ifndef LEAN
	parallel_for(blocked_range<size_t>(0, c.graph.nrVertices), SumVertexCoordinates(c.graph.coordinates.begin(), kappaInv.begin(), permute.begin(), fine.coordinates.begin()));
#endif
	
	//Create new neighbours lists for the coarse graph.
	parallel_for(blocked_range<size_t>(0, c.graph.nrVertices), SumNeighbourRanges(mu.begin(), kappaInv.begin(), permute.begin(), fine.neighbourRanges.begin()));
	parallel_inclusive_scan(mu.begin(), mu.begin() + c.graph.nrVertices, sigma.begin());
	//Turn the inclusive scan into an exclusive scan.
	sigma.insert(sigma.begin(), 0);
	c.graph.neighbours.resize(sigma[c.graph.nrVertices]);
	//Copy all neighbours.
	parallel_for(blocked_range<size_t>(0, c.graph.nrVertices), CopyNeighbours(c.graph.neighbours.begin(), sigma.begin(), kappaInv.begin(), permute.begin(), fine.neighbourRanges.begin(), fine.neighbours.begin()));
	//Apply kappa to all neighbours.
	parallel_for(blocked_range<size_t>(0, c.graph.nrVertices), ProjectNeighbours(c.graph.neighbours.begin(), sigma.begin(), c.kappa.begin()));
	//Compress neighbour list.
	parallel_for(blocked_range<size_t>(0, c.graph.nrVertices), CompressNeighbours(c.graph.neighbourRanges.begin(), c.graph.neighbours.begin(), sigma.begin()));
	
	return c;
}


