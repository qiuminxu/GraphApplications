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

#include <tbb/tbb.h>

#include "tbbext.h"
#include "graph.h"
#include "matchtbb.h"

using namespace std;
using namespace tbb;
using namespace clu;

//==== Matching TBB classes ====
#define LEFTROTATE(a, b) (((a) << (b)) | ((a) >> (32 - (b))))

//FIXME: Ugly way to stop the matching process.
bool keepMatchingTBB = false;

//Use TEA algorithm, 'GPU Random Numbers via the Tiny Encryption Algorithm', Zafar (2010).
class ColourVertices
{
	public:
		ColourVertices(int * const, const int &);
		~ColourVertices();
		
		void setSeed(const unsigned int &);
		void operator () (const blocked_range<int> &) const;
	
	private:
		int * const match;
		const int nrVertices;
		unsigned int random;
};

ColourVertices::ColourVertices(int * const _match, const int &_nrVertices) :
	match(_match),
	nrVertices(_nrVertices),
	random(12345)
{
	
}

ColourVertices::~ColourVertices()
{

}

void ColourVertices::setSeed(const unsigned int &_random)
{
	random = _random;
}

void ColourVertices::operator () (const blocked_range<int> &r) const
{
	//Apply selection procedure to each pi-value in parallel.
	for (int i = r.begin(); i != r.end(); ++i)
	{
		if (match[i] < 2)
		{
			//There are still vertices to be matched.
			keepMatchingTBB = true;
			
			//Start hashing.
			unsigned int sum = 0, v0 = i, v1 = random;
			
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
	}
}

//Matching class.
class MatchVertices
{
	public:
		MatchVertices(int * const, const int * const, const int &);
		~MatchVertices();
		
		void operator () (const blocked_range<int> &) const;
	
	private:
		int * const match;
		const int * const requests;
		const int nrVertices;
};

MatchVertices::MatchVertices(int * const _match, const int * const _requests, const int &_nrVertices) :
	match(_match),
	requests(_requests),
	nrVertices(_nrVertices)
{

}

MatchVertices::~MatchVertices()
{

}

void MatchVertices::operator () (const blocked_range<int> &s) const
{
	//Match all compatible requests in parallel.
	for (int i = s.begin(); i != s.end(); ++i)
	{
		const int r = requests[i];

		//Only unmatched vertices make requests.
		if (r == nrVertices + 1)
		{
			//This is a vertex without any available neighbours, discard it.
			match[i] = 2;
		}
		else if (r < nrVertices)
		{
			//This vertex has made a valid request.
			if (requests[r] == i)
			{
				//Match the vertices if the request was mutual.
				match[i] = MATCH_VAL(i, r);
			}
		}
	}
}

class MatchRequest
{
	public:
		MatchRequest(int * const, const int * const, const int &, const long &, const long &, const int2 * const, const int * const, const int2 * const);
		~MatchRequest();
		
		void operator () (const blocked_range<int> &) const;
	
	private:
		int * const requests;
		const int * const match;
		const int nrVertices;
		const long twoOmega, startWeight;
		const int2 * const neighbourRanges;
		const int * const vertexWeights;
		const int2 * const neighbours;
};

class MatchRespond
{
	public:
		MatchRespond(int * const, const int * const, const int &, const long &, const long &, const int2 * const, const int * const, const int2 * const);
		~MatchRespond();
		
		void operator () (const blocked_range<int> &) const;
	
	private:
		int * const requests;
		const int * const match;
		const int nrVertices;
		const long twoOmega, startWeight;
		const int2 * const neighbourRanges;
		const int * const vertexWeights;
		const int2 * const neighbours;
};

MatchRequest::MatchRequest(int * const _requests, const int * const _match,
			const int &_nrVertices, const long &_Omega, const long &_startWeight,
			const int2 * const _neighbourRanges, const int * const _vertexWeights,
			const int2 * const _neighbours) :
	requests(_requests),
	match(_match),
	nrVertices(_nrVertices),
	twoOmega(2L*_Omega),
	startWeight(_startWeight),
	neighbourRanges(_neighbourRanges),
	vertexWeights(_vertexWeights),
	neighbours(_neighbours)
{

}

MatchRequest::~MatchRequest()
{
	
}

void MatchRequest::operator () (const blocked_range<int> &s) const
{
	//Make requests in parallel.
	for (int i = s.begin(); i != s.end(); ++i)
	{
		//Look at all blue vertices and let them make requests.
		if (match[i] == 0)
		{
			const int2 indices = neighbourRanges[i];
			const long wgt = vertexWeights[i];
			long maxWeight = startWeight;
			int candidate = nrVertices;
			int dead = 1;

			for (int j = indices.x; j < indices.y; ++j)
			{
				const int2 ni = neighbours[j];
				const int nm = match[ni.x];

				//Do we have an unmatched neighbour?
				if (nm < 4)
				{
					//Is this neighbour red?
					if (nm == 1)
					{
						const long weight = twoOmega*(long)ni.y - wgt*(long)vertexWeights[ni.x];
					
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
}

MatchRespond::MatchRespond(int * const _requests, const int * const _match,
			const int &_nrVertices, const long &_Omega, const long &_startWeight,
			const int2 * const _neighbourRanges, const int * const _vertexWeights,
			const int2 * const _neighbours) :
	requests(_requests),
	match(_match),
	nrVertices(_nrVertices),
	twoOmega(2L*_Omega),
	startWeight(_startWeight),
	neighbourRanges(_neighbourRanges),
	vertexWeights(_vertexWeights),
	neighbours(_neighbours)
{

}

MatchRespond::~MatchRespond()
{
	
}

void MatchRespond::operator () (const blocked_range<int> &s) const
{
	//Make requests in parallel.
	for (int i = s.begin(); i != s.end(); ++i)
	{
		//Look at all red vertices.
		if (match[i] == 1)
		{
			const int2 indices = neighbourRanges[i];
			const long wgt = vertexWeights[i];
			long maxWeight = startWeight;
			int candidate = nrVertices;

			for (int j = indices.x; j < indices.y; ++j)
			{
				const int2 ni = neighbours[j];
				
				if (match[ni.x] == 0)
				{
					const long weight = twoOmega*(long)ni.y - wgt*(long)vertexWeights[ni.x];
					
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

			if (candidate < nrVertices) requests[i] = candidate;
		}
	}
}

GraphMatchingTBB::GraphMatchingTBB(const bool &_onlyImprove) :
	onlyImprove(_onlyImprove)
{

}

GraphMatchingTBB::~GraphMatchingTBB()
{

}

vector<int> GraphMatchingTBB::match(const Graph &graph) const
{
	//Assumes that the order of the vertices has already been randomized.
	vector<int> mu(graph.nrVertices, 0);
	
	//Create requests array.
	vector<int> requests(graph.nrVertices, 0);
	
	//Initialise kernels.
	ColourVertices colour(&mu[0], graph.nrVertices);
	MatchRequest requester(&requests[0], &mu[0], graph.nrVertices, graph.Omega, (onlyImprove ? 0 : -graph.Omega*graph.Omega), &graph.neighbourRanges[0], &graph.vertexWeights[0], &graph.neighbours[0]);
	MatchRespond responder(&requests[0], &mu[0], graph.nrVertices, graph.Omega, (onlyImprove ? 0 : -graph.Omega*graph.Omega), &graph.neighbourRanges[0], &graph.vertexWeights[0], &graph.neighbours[0]);
	MatchVertices matcher(&mu[0], &requests[0], graph.nrVertices);
	const blocked_range<int> range(0, graph.nrVertices);
	
	int count = 0;
	keepMatchingTBB = true;
	
	while (keepMatchingTBB && ++count < 32)
	{
		keepMatchingTBB = false;

		colour.setSeed(rand());
		
		parallel_for(range, colour);
		parallel_for(range, requester);
		parallel_for(range, responder);
		parallel_for(range, matcher);
	}
	
#ifndef NDEBUG
	int matchCount = 0;
	
	for (int i = range.begin(); i != range.end(); ++i)
	{
		if (MATCH_MATCHED(mu[i])) ++matchCount;
	}
	
	cerr << "Matched " << matchCount << "/" << graph.nrVertices << " vertices (TBB)." << endl;
#endif
	
	return mu;
}

class MarkSatellites
{
	public:
		MarkSatellites(int * const, const int * const, const int &, const long &, const int2 * const, const int2 * const);
		~MarkSatellites();
		
		void operator () (const blocked_range<int> &) const;
	
	private:
		int * const marks;
		const int * const match;
		const int nrVertices;
		const long twoOmega;
		const int2 * const neighbourRanges;
		const int2 * const neighbours;
};

MarkSatellites::MarkSatellites(int * const _marks, const int * const _match,
			const int &_nrVertices, const long &_Omega,
			const int2 * const _neighbourRanges,
			const int2 * const _neighbours) :
	marks(_marks),
	match(_match),
	nrVertices(_nrVertices),
	twoOmega(2L*_Omega),
	neighbourRanges(_neighbourRanges),
	neighbours(_neighbours)
{

}

MarkSatellites::~MarkSatellites()
{
	
}

void MarkSatellites::operator () (const blocked_range<int> &s) const
{
	for (int i = s.begin(); i != s.end(); ++i)
	{
		//Look at all unmatched vertices and determine whether they will become satellites.
		if (!MATCH_MATCHED(match[i]))
		{
			const int2 indices = neighbourRanges[i];
			const long wgt = indices.y - indices.x;
			long neighbourDegrees = 0;

			for (int j = indices.x; j < indices.y; ++j)
			{
				const int2 ni = neighbours[j];
				const int2 nindices = neighbourRanges[ni.x];
				
				neighbourDegrees += nindices.y - nindices.x;
			}

			marks[i] = (MATCH_SATELLITE*wgt*wgt <= neighbourDegrees ? 1 : 0);
		}
		else
		{
			marks[i] = 0;
		}
	}
}

class MatchSatellites
{
	public:
		MatchSatellites(int * const, const int * const, const int &, const long &, const long &, const int2 * const, const int * const, const int2 * const);
		~MatchSatellites();
		
		void operator () (const blocked_range<int> &) const;
	
	private:
		int * const match;
		const int * const marks;
		const int nrVertices;
		const long twoOmega, startWeight;
		const int2 * const neighbourRanges;
		const int * const vertexWeights;
		const int2 * const neighbours;
};

MatchSatellites::MatchSatellites(int * const _match, const int * const _marks,
			const int &_nrVertices, const long &_Omega, const long &_startWeight,
			const int2 * const _neighbourRanges, const int * const _vertexWeights,
			const int2 * const _neighbours) :
	match(_match),
	marks(_marks),
	nrVertices(_nrVertices),
	twoOmega(2L*_Omega),
	startWeight(_startWeight),
	neighbourRanges(_neighbourRanges),
	vertexWeights(_vertexWeights),
	neighbours(_neighbours)
{

}

MatchSatellites::~MatchSatellites()
{
	
}

void MatchSatellites::operator () (const blocked_range<int> &s) const
{
	for (int i = s.begin(); i != s.end(); ++i)
	{
		//Look at all satellites and match them.
		if (marks[i] == 1)
		{
			const int2 indices = neighbourRanges[i];
			const long wgt = vertexWeights[i];
			long maxWeight = startWeight;
			int candidate = nrVertices;

			for (int j = indices.x; j < indices.y; ++j)
			{
				const int2 ni = neighbours[j];
				
				if (marks[ni.x] != 1)
				{
					//Match to the heaviest non-satellite neighbour.
					const long weight = twoOmega*(long)ni.y - wgt*(long)vertexWeights[ni.x];
				
					if (weight > maxWeight)
					{
						maxWeight = weight;
						candidate = ni.x;
					}
				}
			}
			
			if (candidate < nrVertices) match[i] = match[candidate];
		}
	}
}

void GraphMatchingTBB::matchSatellites(vector<int> &mu, const Graph &graph) const
{
	//Match satellite vertices to their most favourable non-satellite neighbour.
	assert((int)mu.size() == graph.nrVertices);
	
	vector<int> marks(mu.size());
	MarkSatellites marker(&marks[0], &mu[0], graph.nrVertices, graph.Omega, &graph.neighbourRanges[0], &graph.neighbours[0]);
	MatchSatellites matcher(&mu[0], &marks[0], graph.nrVertices, graph.Omega, (onlyImprove ? 0 : -graph.Omega*graph.Omega), &graph.neighbourRanges[0], &graph.vertexWeights[0], &graph.neighbours[0]);
	const blocked_range<int> range(0, graph.nrVertices);
	
	parallel_for(range, marker);
	parallel_for(range, matcher);

#ifndef NDEBUG
	int nrSatellites = parallel_reduce(marks.begin(), marks.end());
	
	cerr << "Identified " << nrSatellites << "/" << graph.nrVertices << " (" << (100*nrSatellites)/graph.nrVertices << "%) vertices as satellites." << endl;
#endif
}

