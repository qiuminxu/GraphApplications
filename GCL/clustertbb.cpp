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
#include <cmath>
#include <climits>
#include <algorithm>

#include <tbb/tbb.h>

#include "tbbext.h"
#include "clustertbb.h"

using namespace clu;
using namespace std;
using namespace tbb;

class SumModularities
{
	public:
		SumModularities(const Graph &_graph) : sum(0), graph(_graph) {};
		SumModularities(const SumModularities &a, split) : sum(0), graph(a.graph) {};
		~SumModularities() {};
		
		void join(const SumModularities &a)
		{
			sum += a.sum;
		};
		
		void operator () (const blocked_range<int> &r)
		{
			long _sum = 0;
			
			for (int i = r.begin(); i != r.end(); ++i)
			{
				const int2 indices = graph.neighbourRanges[i];
				const long wgt = graph.vertexWeights[i];
				long extWgt = 0;
				
				for (int j = indices.x; j < indices.y; ++j)
				{
					const int2 ni = graph.neighbours[j];
					
					extWgt += (long)ni.y;
				}
				
				_sum += wgt*(2L*graph.Omega - wgt) - 2L*graph.Omega*extWgt;
			}
			
			sum += _sum;
		};
		
		long sum;
		
	private:
		const Graph &graph;
};

class GetClusterMod
{
	public:
		GetClusterMod(const Graph &_graph) : graph(_graph) {};
		~GetClusterMod() {};
		
		void operator () (const blocked_range<int> &r) const
		{
			for (int i = r.begin(); i != r.end(); ++i)
			{
				//Calculate contributions of individual clusters to the modularity.
			}
		};
		
	private:
		const Graph &graph;
};

ClusterTBB::ClusterTBB() :
	Cluster()
{

}

ClusterTBB::~ClusterTBB()
{

}

vector<int> ClusterTBB::cluster(const Graph &graph, const double &quality, Drawer *drawer) const
{
#ifdef TIME
	tick_count clusterStart = tick_count::now();
	double matchTime = 0.0, coarsenTime = 0.0, totalTime = 0.0;
#endif
	
	//Draw the graph in its original state.
	if (drawer)
	{
		drawer->drawGraphMatrix(graph);
		drawer->drawGraphCoordinates(graph);
	}
	
	//Coarsen the graph recursively by merging clusters whose merge would increase modularity.
	GraphCoarseningTBB coarsen;
	GraphMatchingTBB match1(true), match2(false);
	
	vector<int> clustering(graph.nrVertices), bestClustering(graph.nrVertices);
	
	//Assign initial clustering indices.
	parallel_sequence(clustering.begin(), clustering.end());
	
	int lastNrVertices = graph.nrVertices;
	bool isImproved = true;
	const Graph *cur = &graph;
	CoarsenedGraph tmp;
	long maxMod = LONG_MIN;
	int nrLevels = 0;
	bool done = false;
	
	do
	{
		//Stop coarsening if the modularity cannot further be increased.
		lastNrVertices = cur->nrVertices;
		
		//Update modularity.
		SumModularities mods(*cur);
		
		parallel_reduce(blocked_range<int>(0, cur->nrVertices), mods);
		
		//Check modularity.
#ifndef NDEBUG
		cerr << (double)mods.sum/(double)(4L*graph.Omega*graph.Omega) << " == " << modularity(graph, clustering) << " (" << mods.sum << ") (used only-increase matching " << isImproved << ")" << endl;
		assert(fabs(((double)mods.sum/(double)(4L*graph.Omega*graph.Omega)) - modularity(graph, clustering)) < 1.0e-8);
#endif
		
		//Keep track of best clustering so far.
		if (mods.sum >= maxMod)
		{
			maxMod = mods.sum;
			bestClustering = clustering;
		}
		else if (mods.sum < CLUSTER_STOPFRAC*maxMod)
		{
			//If we are moving away from the maximum, stop clustering.
			done = true;
		}

#ifdef TIME
		tick_count matchStart = tick_count::now();
#endif
		vector<int> mu = (isImproved ? match1.match(*cur) : match2.match(*cur));
		
		if (isImproved) match1.matchSatellites(mu, *cur);
		else match2.matchSatellites(mu, *cur);
		
#ifdef TIME
		matchTime += (tick_count::now() - matchStart).seconds();
#endif
		
#ifdef TIME
		tick_count coarsenStart = tick_count::now();
#endif
		tmp = coarsen.coarsen(*cur, mu);
#ifdef TIME
		coarsenTime += (tick_count::now() - coarsenStart).seconds();
#endif
		
		//Apply coarsening to clusterings.
		parallel_gather(clustering.begin(), clustering.end(), tmp.kappa.begin(), clustering.begin());
		
		cur = &tmp.graph;
		++nrLevels;
		
#ifndef LEAN
		//Draw intermediate clustering.
		if (drawer)
		{
			drawer->drawGraphMatrix(*cur);
			drawer->drawGraphCoordinates(*cur);
		}
#endif
		
		if (!isImproved && cur->nrVertices >= lastNrVertices) break;
		
		isImproved = (cur->nrVertices < quality*lastNrVertices);
	}
	while (!done);
	
	//Draw the graph final clustering.
	if (drawer)
	{
		drawer->drawGraphMatrix(graph);
		drawer->drawGraphClustering(graph, bestClustering);
	}
	
#ifndef NDEBUG
	cerr << "Created a clustering of " << graph.nrVertices << " vertices into " << lastNrVertices << " components in " << nrLevels << " iterations (TBB)." << endl;
#endif
#ifdef TIME
	totalTime = (tick_count::now() - clusterStart).seconds();
	cerr << scientific << "Clustering took " << totalTime << "s of which " << matchTime << "s (" << (int)floor(100.0*matchTime/totalTime) <<"%) was spent matching and " << coarsenTime << "s (" << (int)floor(100.0*coarsenTime/totalTime) << "%) coarsening." << endl;
#endif
	
	return bestClustering;
}

