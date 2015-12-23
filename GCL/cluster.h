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
#ifndef CLUSTER_CLUSTER_H
#define CLUSTER_CLUSTER_H

#include <vector>

#include "graph.h"
#include "match.h"
#include "coarsen.h"
#include "vis.h"

namespace clu
{

#define CLUSTER_STOPFRAC 0.95

class Cluster
{
	public:
		Cluster();
		virtual ~Cluster();
		
		virtual std::vector<int> cluster(const Graph &, const double & = 1.0, Drawer * = 0) const = 0;
		static double modularity(const Graph &, const std::vector<int> &);
};

}

#endif

