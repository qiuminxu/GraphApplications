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
#include <list>
#include <map>
#include <queue>
#include <algorithm>

#include <cassert>

#include "matchcpu.h"

using namespace std;
using namespace mtc;

GraphMatching::GraphMatching(const Graph &_graph) :
		graph(_graph)
{
	/*
	if (graph.empty())
	{
		cerr << "Empty graphs cannot be matched!" << endl;
		throw exception();
	}
	*/
}

GraphMatching::~GraphMatching()
{
	
}

vector<int> GraphMatching::initialMatching() const
{
	/*
	if (graph.empty())
	{
		cerr << "Empty graph!" << endl;
		throw exception();
	}
	*/

	vector<int> match(graph.nrVertices, 0);

	return match;
}

void GraphMatching::getWeight(double &_weight, long &_size, const vector<int> &match, const Graph &graph)
{
	assert((int)match.size() == graph.nrVertices);

	double weight = 0.0;
	long size = 0;

	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int m = match[i];

		if (isMatched(m))
		{
			//This vertex has been matched, find out which of its neighbours it has been matched to.
			const int2 indices = graph.neighbourRanges[i];

			for (int j = indices.x; j < indices.y; ++j)
			{
				const int k = graph.neighbours[j];

				if (match[k] == m)
				{
					weight += (double)graph.neighbourWeights[j];
					size += 1;
					break;
				}
			}
		}
	}

	//We counted the weights double.
	_weight = weight/2.0;
	_size = size;
}

class SortVerticesMatch
{
	public:
		SortVerticesMatch(const vector<int> &_v) :
			v(_v)
		{
			
		};

		~SortVerticesMatch()
		{
			
		};

		inline bool operator()(int a, int b) const
		{
			const bool ma = GraphMatching::isMatched(v[a]), mb = GraphMatching::isMatched(v[b]);

			if (ma && !mb) return false;
			else if (!ma && mb) return true;

			return v[a] < v[b];
		};

	private:
		const vector<int> &v;
};

bool GraphMatching::testMatching(const vector<int> &match, const Graph &graph)
{
	if ((int)match.size() != graph.nrVertices)
	{
		cerr << "Incompatible matching size!" << endl;
		return false;
	}

	//Sort all vertices by their matching index.
	SortVerticesMatch sortByMatching(match);
	vector<int> order(graph.nrVertices);

	for (int i = 0; i < graph.nrVertices; ++i) order[i] = i;

	sort(order.begin(), order.end(), sortByMatching);

	//Find all unmatched vertices.
	vector<int>::const_iterator oi = order.begin();
	int nrUnMatched = 0;

	while (oi != order.end())
	{
		if (isMatched(match[*oi])) break;

		++nrUnMatched;
		++oi;
	}

	//Find the number of vertices that have been matched to more than one other vertex.
	//Also verify that all vertices have been matched to a neighbour.
	int nrUnderMatched = 0, nrMultiMatched = 0, nrInvalidlyMatched = 0;

	while (oi != order.end())
	{
		const int i1 = *oi;
		const int curMatch = match[i1];

		assert(isMatched(curMatch));

		++oi;
		
		//If we only have a single vertex with this matching index, this is not good.
		if (oi == order.end())
		{
			++nrUnderMatched;
			continue;
		}
		
		//Look at the next vertex.
		const int i2 = *oi;

		++oi;

		if (match[i2] != curMatch)
		{
			++nrUnderMatched;
			continue;
		}

		//Check that the vertices have actually been matched to neighbours.
		bool found1 = false, found2 = false;
		const int2 ni1 = graph.neighbourRanges[i1], ni2 = graph.neighbourRanges[i2];

		for (int i = ni1.x; i < ni1.y && !found1; ++i)
		{
			if (graph.neighbours[i] == i2) found1 = true;
		}

		for (int i = ni2.x; i < ni2.y && !found2; ++i)
		{
			if (graph.neighbours[i] == i1) found2 = true;
		}

		if (!found1 || !found2)
		{
			++nrInvalidlyMatched;
		}

		//If too many vertices have been matched, this is bad too.
		while (oi != order.end())
		{
			if (match[*oi] != curMatch) break;

			++nrMultiMatched;
			++oi;
		}
	}
	
	//Count the number of vertices that can still be matched.
	int nrMatchable = 0;

	for (int i = 0; i < (int)match.size(); ++i)
	{
		int validNeighbours = 0;

		if (isMatched(match[i])) continue;

		const int2 indices = graph.neighbourRanges[i];

		for (int j = indices.x; j < indices.y; ++j)
		{
			const int ni = graph.neighbours[j];

			//We never match to ourselves.
			if (ni == i) continue;

			if (!isMatched(match[ni])) ++validNeighbours;
		}

		if (validNeighbours > 0) ++nrMatchable;
	}

	cout << "Verified a matching of " << match.size() << " vertices, of these" << endl;
	cout << "    " << nrUnderMatched << " were matched to themselves," << endl;
	cout << "    " << nrMultiMatched << " were matched to more than one vertex," << endl;
	cout << "    " << nrInvalidlyMatched << " were not matched to a neighbour," << endl;
	cout << "    " << nrMatchable << " can still be matched." << endl;

	return (nrUnderMatched == 0 && nrMultiMatched == 0 && nrInvalidlyMatched == 0);
}

GraphMatchingCPURandom::GraphMatchingCPURandom(const Graph &_graph) :
	GraphMatching(_graph)
{

}

GraphMatchingCPURandom::~GraphMatchingCPURandom()
{
	
}

void GraphMatchingCPURandom::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2) const
{
	//This is a random greedy matching algorithm.
	//Assumes that the order of the vertices has already been randomized.
	
	assert((int)match.size() == graph.nrVertices);
	
	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);

	for (int i = 0; i < graph.nrVertices; ++i)
	{
		//Check if this vertex is unmatched.
		if (!isMatched(match[i]))
		{
			//If so, match it to an unmatched neighbour.
			const int2 indices = graph.neighbourRanges[i];
			int candidate = graph.nrVertices;
			
			for (int j = indices.x; j < indices.y; ++j)
			{
				const int ni = graph.neighbours[j];
				
				if (!isMatched(match[ni]) && ni != i)
				{
					candidate = ni;
					break;
				}
			}
			
			//Have we found an unmatched neighbour?
			if (candidate < graph.nrVertices)
			{
				//Match it.
				const int m = matchVal(i, candidate);
				
				match[i] = m;
				match[candidate] = m;
			}
		}
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);
}

GraphMatchingCPUMinDeg::GraphMatchingCPUMinDeg(const Graph &_graph) :
	GraphMatching(_graph)
{

}

GraphMatchingCPUMinDeg::~GraphMatchingCPUMinDeg()
{
	
}

void GraphMatchingCPUMinDeg::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2) const
{
	//This is a two-sided dynamic minimum degree greedy matching algorithm.
	//Assumes that no vertices have been matched yet.
	
	assert((int)match.size() == graph.nrVertices);
	
	//Allocate memory.
	vector<multimap<int, int>::iterator> mapPtrs(graph.nrVertices);
	
	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);
	
	//Add unmatched vertices, ordered by their degrees.
	multimap<int, int> orderedVertices;
	
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int2 indices = graph.neighbourRanges[i];
		
		//We need to keep track of the positions of the vertices in the map to be able to delete them efficiently.
		if (indices.y > indices.x) mapPtrs[i] = orderedVertices.insert(pair<int, int>(indices.y - indices.x, i));
	}
	
	//Repeatedly match vertices of lowest degree.
	while (!orderedVertices.empty())
	{
		//Retrieve current vertex (with lowest degree).
		const int i = orderedVertices.begin()->second;
		const int2 indices = graph.neighbourRanges[i];

		//This vertex cannot be matched already.
		assert(!isMatched(match[i]));
		
		//If this vertex can no longer be matched, remove it immediately.
		if (orderedVertices.begin()->first == 0)
		{
			orderedVertices.erase(mapPtrs[i]);
			continue;
		}
		
		//Find neighbour with minimal degree (two-sided).
		int minDegree = graph.nrVertices + 1;
		int candidate = graph.nrVertices;
	
		for (int j = indices.x; j < indices.y; ++j)
		{
			const int ni = graph.neighbours[j];
			
			//Only look at unmatched neighbours.
			if (!isMatched(match[ni]) && ni != i)
			{
				const int degree = mapPtrs[ni]->first;

				if (degree < minDegree)
				{
					minDegree = degree;
					candidate = ni;
				}
			}
		}
		
		//Because i's degree is > 0, we should have found a neighbour.
		assert(candidate < graph.nrVertices);
		
		//Match the vertices, remove them from the map, and update their neighbour's degrees.
		const int m = matchVal(i, candidate);
		
		match[i] = m;
		match[candidate] = m;
		
		//Remove vertices.
		orderedVertices.erase(mapPtrs[candidate]);
		orderedVertices.erase(mapPtrs[i]);
		
		//Change degrees of current neighbours.
		for (int j = indices.x; j < indices.y; ++j)
		{
			const int ni = graph.neighbours[j];
		
			if (!isMatched(match[ni]) && ni != i)
			{
				const int degree = mapPtrs[ni]->first;
				
				orderedVertices.erase(mapPtrs[ni]);
				//TODO: Can we do this more efficiently with some iterator magic?
				if (degree > 1) mapPtrs[ni] = orderedVertices.insert(pair<int, int>(degree - 1, ni));
			}
		}
		
		//Change degrees of candidate neighbours.
		const int2 indices2 = graph.neighbourRanges[candidate];
		
		for (int j = indices2.x; j < indices2.y; ++j)
		{
			const int ni = graph.neighbours[j];
		
			if (!isMatched(match[ni]) && ni != i)
			{
				const int degree = mapPtrs[ni]->first;
				
				orderedVertices.erase(mapPtrs[ni]);
				//TODO: Can we do this more efficiently with some iterator magic?
				if (degree > 1) mapPtrs[ni] = orderedVertices.insert(pair<int, int>(degree - 1, ni));
			}
		}
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);
}

GraphMatchingCPUStatMinDeg::GraphMatchingCPUStatMinDeg(const Graph &_graph) :
	GraphMatching(_graph)
{

}

GraphMatchingCPUStatMinDeg::~GraphMatchingCPUStatMinDeg()
{
	
}

class SortByDegree
{
	public:
		SortByDegree(const Graph &_g) :
			g(_g)
		{
			
		};

		~SortByDegree()
		{
			
		};

		inline bool operator()(int a, int b) const
		{
			return g.neighbourRanges[a].y + g.neighbourRanges[b].x < g.neighbourRanges[b].y + g.neighbourRanges[a].x;
		};

	private:
		const Graph &g;
};

void GraphMatchingCPUStatMinDeg::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2) const
{
	//This is a one-sided static minimum degree greedy matching algorithm.
	
	assert((int)match.size() == graph.nrVertices);
	
	//Create array for all vertex degrees.
	vector<int> order(graph.nrVertices);
	
	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);

	//Sort vertices by their degrees.
	for (int i = 0; i < graph.nrVertices; ++i) order[i] = i;
	
	sort(order.begin(), order.end(), SortByDegree(graph));

	//Treat vertices in order of their degree.
	for (int h = 0; h < graph.nrVertices; ++h)
	{
		const int i = order[h];
		
		//Check if this vertex is unmatched.
		if (!isMatched(match[i]))
		{
			//If so, match it to an unmatched neighbour (one-sided).
			const int2 indices = graph.neighbourRanges[i];
			int candidate = graph.nrVertices;
			
			for (int j = indices.x; j < indices.y; ++j)
			{
				const int ni = graph.neighbours[j];
				
				if (!isMatched(match[ni]) && ni != i)
				{
					candidate = ni;
					break;
				}
			}
			
			//Have we found an unmatched neighbour?
			if (candidate < graph.nrVertices)
			{
				//Match it.
				const int m = matchVal(i, candidate);
				
				match[i] = m;
				match[candidate] = m;
			}
		}
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);
}

GraphMatchingCPUKarpSipser::GraphMatchingCPUKarpSipser(const Graph &_graph) :
	GraphMatching(_graph)
{

}

GraphMatchingCPUKarpSipser::~GraphMatchingCPUKarpSipser()
{
	
}

void GraphMatchingCPUKarpSipser::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2) const
{
	//This is the one-sided Karp-Sipser greedy matching algorithm.
	//Assumes that the order of the vertices has already been randomized and that the given matching is empty.
	
	assert((int)match.size() == graph.nrVertices);
	
	//Allocate memory.
	vector<list<int>::iterator> listPtrs(graph.nrVertices);
	vector<int> degrees(graph.nrVertices);
	
	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);
	
	//Calculate all vertex degrees and create singleton list.
	queue<int> singletons;
	list<int> others;
	
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int2 indices = graph.neighbourRanges[i];
		
		degrees[i] = indices.y - indices.x;
		
		if (degrees[i] == 1) singletons.push(i);
		else if (degrees[i] > 1) listPtrs[i] = others.insert(others.end(), i);
	}
		
	while (!singletons.empty() || !others.empty())
	{
		//First match all singletons.
		while (!singletons.empty())
		{
			const int i = singletons.front();
			
			singletons.pop();
			
			assert(degrees[i] <= 1);
			
			//This vertex may have been matched earlier, if so, discard it.
			if (isMatched(match[i]) || degrees[i] < 1) continue;
			
			//Find its single unmatched neighbour.
			const int2 indices = graph.neighbourRanges[i];
			int candidate = graph.nrVertices;
			
			for (int j = indices.x; j < indices.y; ++j)
			{
				const int ni = graph.neighbours[j];
				
				if (!isMatched(match[ni]) && ni != i)
				{
					candidate = ni;
					break;
				}
			}
			
			//This neighbour should always be available.
			assert(candidate < graph.nrVertices);
			
			//Match it.
			const int m = matchVal(i, candidate);
			
			match[i] = m;
			match[candidate] = m;
			
			//Decrease the degrees of all of the candidate's neighbours and add them to the singleton list when appropriate.
			const int2 indices2 = graph.neighbourRanges[candidate];
			
			for (int j = indices2.x; j < indices2.y; ++j)
			{
				const int ni = graph.neighbours[j];
				
				degrees[ni]--;
				
				if (degrees[ni] == 1 && !isMatched(match[ni]))
				{
					others.erase(listPtrs[ni]);
					singletons.push(ni);
				}
			}
		}
		
		//If no more singletons are available, start random matching until we produce more singletons.
		bool done = false;
		
		while (!others.empty() && !done)
		{
			for (list<int>::iterator h = others.begin(); h != others.end(); h = others.erase(h))
			{
				const int i = *h;
				
				//Remove this vertex if it has already been matched or can no longer be matched.
				if (isMatched(match[i]) || degrees[i] == 0) continue;
				
				assert(degrees[i] > 1);
				
				//Match this vertex to its first available neighbour.
				const int2 indices = graph.neighbourRanges[i];
				int candidate = graph.nrVertices;
				
				for (int j = indices.x; j < indices.y; ++j)
				{
					const int ni = graph.neighbours[j];
					
					if (!isMatched(match[ni]) && ni != i)
					{
						candidate = ni;
						break;
					}
				}
				
				assert(candidate < graph.nrVertices);
				
				//Perform the match.
				const int m = matchVal(i, candidate);
				
				match[i] = m;
				match[candidate] = m;
				
				//Update all neighbour's degrees.
				//If we encounter new singletons, we will start to match these.
				for (int j = indices.x; j < indices.y; ++j)
				{
					const int ni = graph.neighbours[j];
					
					if (--degrees[ni] == 1 && !isMatched(match[ni]))
					{
						others.erase(listPtrs[ni]);
						singletons.push(ni);
						done = true;
					}
				}
				
				const int2 indices2 = graph.neighbourRanges[candidate];
				
				for (int j = indices2.x; j < indices2.y; ++j)
				{
					const int ni = graph.neighbours[j];
					
					if (--degrees[ni] == 1 && !isMatched(match[ni]))
					{
						others.erase(listPtrs[ni]);
						singletons.push(ni);
						done = true;
					}
				}
			}
		}
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);
}

GraphMatchingCPUWeighted::GraphMatchingCPUWeighted(const Graph &_graph) :
	GraphMatching(_graph)
{

}

GraphMatchingCPUWeighted::~GraphMatchingCPUWeighted()
{
	
}

void GraphMatchingCPUWeighted::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2) const
{
	//This is a greedy weighted matching algorithm.
	//Assumes that the order of the vertices has already been randomized.
	
	assert((int)match.size() == graph.nrVertices);
	
	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);

	for (int i = 0; i < graph.nrVertices; ++i)
	{
		//Check if this vertex is unmatched.
		if (!isMatched(match[i]))
		{
			//If so, match it to an unmatched neighbour.
			const int2 indices = graph.neighbourRanges[i];
			float maxWeight = -1.0;
			int candidate = graph.nrVertices;
			
			for (int j = indices.x; j < indices.y; ++j)
			{
				const int ni = graph.neighbours[j];
				
				if (!isMatched(match[ni]) && ni != i)
				{
					const float nw = graph.neighbourWeights[j];
					
					if (nw > maxWeight)
					{
						maxWeight = nw;
						candidate = ni;
					}
				}
			}
			
			//Have we found an unmatched neighbour?
			if (candidate < graph.nrVertices)
			{
				//Match it.
				const int m = matchVal(i, candidate);
				
				match[i] = m;
				match[candidate] = m;
			}
		}
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);
}

class SortEdgesWeight
{
	public:
		SortEdgesWeight()
		{
			
		};

		~SortEdgesWeight()
		{
			
		};

		inline bool operator()(const Edge &a, const Edge &b) const
		{
			return a.w > b.w;
		};
};

GraphMatchingCPUWeightedEdge::GraphMatchingCPUWeightedEdge(const Graph &_graph) :
	GraphMatching(_graph)
{

}

GraphMatchingCPUWeightedEdge::~GraphMatchingCPUWeightedEdge()
{
	
}

void GraphMatchingCPUWeightedEdge::performMatching(vector<int> &match, cudaEvent_t &t1, cudaEvent_t &t2) const
{
	//This is a greedy weighted matching algorithm.
	//Instead of being vertex oriented, this is an edge oriented algorithm.
	vector<Edge> order(graph.nrEdges);
	SortEdgesWeight orderByWeight;
	
	assert((int)match.size() == graph.nrVertices);
	
	for (int i = 0; i < graph.nrEdges; ++i) order[i] = graph.edges[i];
	
	cudaEventRecord(t1, 0);
	cudaEventSynchronize(t1);

	//Sort edges by their weight.
	sort(order.begin(), order.end(), orderByWeight);

	//Start matching vertices, heavy edges first.
	for (vector<Edge>::const_iterator e = order.begin(); e != order.end(); ++e)
	{
		//Check if both vertices are still unmatched.
		if (!isMatched(match[e->x]) && !isMatched(match[e->y]))
		{
			//If so, match them.
			const int m = matchVal(e->x, e->y);

			match[e->x] = m;
			match[e->y] = m;
		}
	}
	
	cudaEventRecord(t2, 0);
	cudaEventSynchronize(t2);
}

