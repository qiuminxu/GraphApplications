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
#include <set>
#include <algorithm>

#include "cluster.h"

using namespace clu;
using namespace std;

Cluster::Cluster()
{

}

Cluster::~Cluster()
{
	
}

double Cluster::modularity(const Graph &graph, const vector<int> &component)
{
	const int nrComponents = *max_element(component.begin(), component.end()) + 1;
	
	if (graph.nrVertices != (int)component.size() || nrComponents <= 0)
	{
		cerr << "Invalid number of vertex components!" << endl;
		return -2.0;
	}
	
#ifndef NDEBUG
	cerr << "Calculating modularity of " << graph.nrVertices << " vertices divided into " << nrComponents << " clusters..." << endl;
#endif
	
	long Omega = 0;
	vector<long> vertexWeights(graph.nrVertices, 0);
	vector<long> edgeWeightSum(nrComponents, 0);
	vector<long> vertexWeightSum(nrComponents, 0);
	
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int2 r = graph.neighbourRanges[i];
		long w = 0;
		
		for (int j = r.x; j < r.y; ++j) w += graph.neighbours[j].y;
		
		vertexWeights[i] = w;
	}
	
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int c = component[i];
		const int2 r = graph.neighbourRanges[i];
		
		vertexWeightSum[c] += vertexWeights[i];
		
		for (int j = r.x; j < r.y; ++j)
		{
			Omega += graph.neighbours[j].y;
			
			//Internal edge?
			if (component[graph.neighbours[j].x] == c)
			{
				edgeWeightSum[c] += graph.neighbours[j].y;
			}
		}
	}
	
	//N.B. all edge weights are double (undirected graph).
	long term1 = 0, term2 = 0;
	
	assert((Omega & 1) == 0);
	Omega /= 2L;
	
	assert(Omega == graph.Omega);
	
	for (int i = 0; i < nrComponents; ++i)
	{
		assert((edgeWeightSum[i] & 1) == 0);
		term1 += edgeWeightSum[i]/2L;
		term2 += vertexWeightSum[i]*vertexWeightSum[i];
	}
	
	return ((double)term1/(double)Omega) - ((double)term2/(double)(4L*Omega*Omega));
}

