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
#ifndef CLUSTER_COARSEN_TBB_H
#define CLUSTER_COARSEN_TBB_H

#include <vector>

#include "graph.h"
#include "coarsen.h"

namespace clu
{

class GraphCoarseningTBB : public GraphCoarsening
{
	public:
		GraphCoarseningTBB();
		~GraphCoarseningTBB();
		
		CoarsenedGraph coarsen(const Graph &, const std::vector<int> &) const;
};

}

#endif

