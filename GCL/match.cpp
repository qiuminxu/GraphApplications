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
#include <algorithm>
#include <cassert>

#include "match.h"

using namespace std;
using namespace clu;

GraphMatching::GraphMatching()
{

}

GraphMatching::~GraphMatching()
{

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
			const bool ma = MATCH_MATCHED(v[a]), mb = MATCH_MATCHED(v[b]);

			if (ma && !mb) return false;
			else if (!ma && mb) return true;

			return v[a] < v[b];
		};

	private:
		const vector<int> &v;
};

bool GraphMatching::test(const vector<int> &mu, const Graph &graph)
{
	if ((int)mu.size() != graph.nrVertices)
	{
		cerr << "Incompatible matching size!" << endl;
		return false;
	}

	//Sort all vertices by their matching index.
	SortVerticesMatch sortByMatching(mu);
	vector<int> order(graph.nrVertices);

	for (int i = 0; i < graph.nrVertices; ++i) order[i] = i;

	sort(order.begin(), order.end(), sortByMatching);

	//Find all unmatched vertices.
	vector<int>::const_iterator oi = order.begin();
	int nrUnMatched = 0;

	while (oi != order.end())
	{
		if (MATCH_MATCHED(mu[*oi])) break;

		++nrUnMatched;
		++oi;
	}

	//Find the number of vertices that have been matched to more than one other vertex.
	//Also verify that all vertices have been matched to a neighbour.
	int nrUnderMatched = 0, nrMultiMatched = 0, nrInvalidlyMatched = 0;

	while (oi != order.end())
	{
		const int i1 = *oi;
		const int curMatch = mu[i1];

		assert(MATCH_MATCHED(curMatch));

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

		if (mu[i2] != curMatch)
		{
			++nrUnderMatched;
			continue;
		}

		//Check that the vertices have actually been matched to neighbours.
		bool found1 = false, found2 = false;
		const int2 ni1 = graph.neighbourRanges[i1], ni2 = graph.neighbourRanges[i2];

		for (int i = ni1.x; i < ni1.y && !found1; ++i)
		{
			if (graph.neighbours[i].x == i2) found1 = true;
		}

		for (int i = ni2.x; i < ni2.y && !found2; ++i)
		{
			if (graph.neighbours[i].x == i1) found2 = true;
		}

		if (!found1 || !found2)
		{
			++nrInvalidlyMatched;
		}

		//If too many vertices have been matched, this is bad too.
		while (oi != order.end())
		{
			if (mu[*oi] != curMatch) break;

			++nrMultiMatched;
			++oi;
		}
	}
	
	//Count the number of vertices that can still be matched.
	int nrMatchable = 0;

	for (int i = 0; i < (int)mu.size(); ++i)
	{
		int validNeighbours = 0;

		if (MATCH_MATCHED(mu[i])) continue;

		const int2 indices = graph.neighbourRanges[i];

		for (int j = indices.x; j < indices.y; ++j)
		{
			const int2 ni = graph.neighbours[j];

			//We never match to ourselves.
			if (ni.x == i) continue;

			if (!MATCH_MATCHED(mu[ni.x])) ++validNeighbours;
		}

		if (validNeighbours > 0) ++nrMatchable;
	}

	cout << "Verified a matching of " << mu.size() << " vertices, of these" << endl;
	cout << "    " << nrUnMatched << " were unmatched (" << (100L*(mu.size() - nrUnMatched))/mu.size() << "% matched)," << endl;
	cout << "    " << nrUnderMatched << " were matched to themselves," << endl;
	cout << "    " << nrMultiMatched << " were matched to more than one vertex," << endl;
	cout << "    " << nrInvalidlyMatched << " were not matched to a neighbour," << endl;
	cout << "    " << nrMatchable << " can still be matched." << endl;

	return (nrUnderMatched == 0 && nrMultiMatched == 0 && nrInvalidlyMatched == 0);
}

GraphMatchingSerial::GraphMatchingSerial() :
	GraphMatching()
{

}

GraphMatchingSerial::~GraphMatchingSerial()
{

}

vector<int> GraphMatchingSerial::match(const Graph &graph) const
{
	//Generate a greedy weighted matching serially.
	vector<int> mu(graph.nrVertices, 0);
	const long twoOmega = 2L*graph.Omega;
	int nrMatched = 0;
	
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		if (!MATCH_MATCHED(mu[i]))
		{
			const int2 indices = graph.neighbourRanges[i];
			const long wgt = graph.vertexWeights[i];
			long maxWeight = -graph.Omega*graph.Omega;
			int candidate = graph.nrVertices;
			
			//Look at all neighbours.
			for (int j = indices.x; j < indices.y; ++j)
			{
				const int2 ni = graph.neighbours[j];
				
				if (!MATCH_MATCHED(mu[ni.x]))
				{
					const long weight = twoOmega*(long)ni.y - wgt*(long)graph.vertexWeights[ni.x];
					
					if (weight > maxWeight)
					{
						maxWeight = weight;
						candidate = ni.x;
					}
				}
			}
			
			//We have found a heavy candidate.
			if (candidate < graph.nrVertices)
			{
				const int m = MATCH_VAL(i, candidate);
				
				mu[i] = m;
				mu[candidate] = m;
				nrMatched += 2;
			}
		}
	}
	
#ifndef NDEBUG
	assert(test(mu, graph));
	cerr << "Matched " << nrMatched << "/" << graph.nrVertices << " vertices serially." << endl;
#endif
	
	return mu;
}


