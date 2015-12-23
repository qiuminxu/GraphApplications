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
#ifndef MATCH_GRAPH_H
#define MATCH_GRAPH_H

#include <iostream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>

namespace mtc
{

class Edge
{
	public:
		Edge();
		Edge(const int &, const int &, const float & = 1.0);
		~Edge();

		int x, y;
		float w;
};

class Graph
{
	public:
		Graph();
		~Graph();

		void clear();
		std::istream &readMETIS(std::istream &);
		std::istream &readMatrixMarket(std::istream &);
		std::vector<int> random_shuffle();
		bool empty() const;

	public:
		int nrVertices, nrVertexWeights, nrEdges;
		std::vector<int> vertexWeights;
		std::vector<int2> neighbourRanges;
		std::vector<int> neighbours;
		std::vector<float> neighbourWeights;
		std::vector<Edge> edges;
};

}

#endif

