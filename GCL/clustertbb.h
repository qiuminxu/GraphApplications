#ifndef CLUSTER_CLUSTER_TBB_H
#define CLUSTER_CLUSTER_TBB_H

#include <vector>

#include "graph.h"
#include "cluster.h"
#include "matchtbb.h"
#include "coarsentbb.h"
#include "vis.h"

namespace clu
{

class ClusterTBB : public Cluster
{
	public:
		ClusterTBB();
		~ClusterTBB();
		
		std::vector<int> cluster(const Graph &, const double & = 1.0, Drawer * = 0) const;
};

}

#endif

