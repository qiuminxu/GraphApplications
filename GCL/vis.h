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
#ifndef CLUSTER_VIS_H
#define CLUSTER_VIS_H

#include <vector>

#include "graph.h"

namespace clu
{

class Drawer
{
	public:
		Drawer();
		virtual ~Drawer();
		
		virtual void drawGraphMatrix(const Graph &) = 0;
		virtual void drawGraphMatrixPermuted(const Graph &, const std::vector<int> &) = 0;
		virtual void drawGraphMatrixPermutedClustering(const Graph &, const std::vector<int> &, const std::vector<int> &) = 0;
		virtual void drawGraphCoordinates(const Graph &) = 0;
		virtual void drawGraphClustering(const Graph &, const std::vector<int> &) = 0;
};

}

#endif

