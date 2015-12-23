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
#ifndef MATCH_MATCH_H
#define MATCH_MATCH_H

#include <vector>

#include "graph.h"

#define NR_MATCH_ROUNDS 20
#define NR_MAX_MATCH_ROUNDS 256
//#define MATCH_INTERMEDIATE_COUNT

namespace mtc
{

class GraphMatching
{
	public:
		GraphMatching(const Graph &);
		virtual ~GraphMatching();

		static void getWeight(double &, long &, const std::vector<int> &, const Graph &);
		static bool testMatching(const std::vector<int> &, const Graph &);
		
		static inline int matchVal(const int &i, const int &j) {return 4 + (i < j ? i : j);};
		static inline bool isMatched(const int &m) {return m >= 4;};
	
		std::vector<int> initialMatching() const;
		virtual void performMatching(std::vector<int> &, cudaEvent_t &, cudaEvent_t &) const = 0;

	protected:
		const Graph &graph;
};

class GraphMatchingCPURandom : public GraphMatching
{
	public:
		GraphMatchingCPURandom(const Graph &);
		~GraphMatchingCPURandom();
		
		void performMatching(std::vector<int> &, cudaEvent_t &, cudaEvent_t &) const;
};

class GraphMatchingCPUMinDeg : public GraphMatching
{
	public:
		GraphMatchingCPUMinDeg(const Graph &);
		~GraphMatchingCPUMinDeg();
		
		void performMatching(std::vector<int> &, cudaEvent_t &, cudaEvent_t &) const;
};

class GraphMatchingCPUStatMinDeg : public GraphMatching
{
	public:
		GraphMatchingCPUStatMinDeg(const Graph &);
		~GraphMatchingCPUStatMinDeg();
		
		void performMatching(std::vector<int> &, cudaEvent_t &, cudaEvent_t &) const;
};

class GraphMatchingCPUKarpSipser : public GraphMatching
{
	public:
		GraphMatchingCPUKarpSipser(const Graph &);
		~GraphMatchingCPUKarpSipser();
		
		void performMatching(std::vector<int> &, cudaEvent_t &, cudaEvent_t &) const;
};

class GraphMatchingCPUWeighted : public GraphMatching
{
	public:
		GraphMatchingCPUWeighted(const Graph &);
		~GraphMatchingCPUWeighted();
		
		void performMatching(std::vector<int> &, cudaEvent_t &, cudaEvent_t &) const;
};

class GraphMatchingCPUWeightedEdge : public GraphMatching
{
	public:
		GraphMatchingCPUWeightedEdge(const Graph &);
		~GraphMatchingCPUWeightedEdge();
		
		void performMatching(std::vector<int> &, cudaEvent_t &, cudaEvent_t &) const;
};

};

#endif
