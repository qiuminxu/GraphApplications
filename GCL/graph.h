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
#ifndef CLUSTER_GRAPH_H
#define CLUSTER_GRAPH_H

#include <iostream>
#include <vector>

#include <vector_functions.h>
#include <vector_types.h>

namespace clu
{

class Graph
{
	public:
		Graph();
		~Graph();
		
		static Graph createStarGraph(const int &);
		static Graph createTreeGraph(const int &, const int &);
		static Graph createGridGraph(const int &, const int &);
		static Graph createLineGraph(const int &);

		void clear();
		std::istream &readMETIS(std::istream &);
		std::istream &readCoordinates(std::istream &);
		std::vector<int> random_shuffle();
		bool empty() const;

	public:
		int nrVertices, nrEdges;
		std::vector<int2> neighbourRanges;
		std::vector<int> vertexWeights;
		std::vector<int2> neighbours;
		std::vector<float2> coordinates;
		//Sum of all edge weights.
		long Omega;
		
	private:
		void setClusterWeights();
};

}

#endif

