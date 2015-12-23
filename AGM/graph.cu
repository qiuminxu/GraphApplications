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
#include <iostream>
#include <exception>
#include <string>
#include <sstream>
#include <algorithm>
#include <vector>
#include <set>
#include <climits>

#include "graph.h"

using namespace std;
using namespace mtc;

Edge::Edge()
{

}

Edge::Edge(const int &_x, const int &_y, const float &_w) :
	x(_x),
	y(_y),
	w(_w)
{

}

Edge::~Edge()
{

}

Graph::Graph() :
	nrVertices(0),
	nrVertexWeights(0),
	nrEdges(0),
	neighbourRanges(),
	vertexWeights(),
	neighbours(),
	neighbourWeights(),
	edges()
{

}

Graph::~Graph()
{

}

bool Graph::empty() const
{
	if (nrVertices == 0 || nrEdges == 0 || neighbours.empty() || neighbourRanges.empty() || edges.empty()) return true;

	return false;
}

void Graph::clear()
{
	nrVertices = 0;
	nrVertexWeights = 0;
	nrEdges = 0;
	neighbourRanges.clear();
	vertexWeights.clear();
	neighbours.clear();
	neighbourWeights.clear();
	edges.clear();
}

istream &Graph::readMatrixMarket(istream &in)
{
	//Read matrix market file from input stream.
	enum {NoStorage, Coordinate, Array} storageType = NoStorage;
	enum {NoData, Real, Complex, Integer, Pattern} dataType = NoData;
	enum {NoLayout, General, Symmetric, Skew, Hermitean} layout = NoLayout;
	bool readDimensions = false;
	int currentRow = 1, currentCol = 1, nrRows = 0, nrCols = 0, nrEntries = 0;
	
	clear();
	
	while (in.good())
	{
		//Read lines from file.
		string line;
		
		getline(in, line);

		while (!line.empty() && isspace(line[0])) line.erase(0, 1);
		
		if (line.length() < 2) continue;
		
		istringstream sline(line);
		
		if (line[0] == '%' && line[1] == '%')
		{
			//We are at the file type definition line.
			while (sline.good())
			{
				string word;
				
				sline >> word;
				
				if (word == "%%MatrixMarket")
				{
					
				}
				else if (word == "matrix")
				{
					
				}
				else if (word == "coordinate")
				{
					storageType = Coordinate;
				}
				else if (word == "array")
				{
					storageType = Array;
				}
				else if (word == "float" || word == "real" || word == "double")
				{
					dataType = Real;
				}
				else if (word == "complex")
				{
					dataType = Complex;
				}
				else if (word == "int" || word == "integer")
				{
					dataType = Integer;
				}
				else if (word == "pattern")
				{
					dataType = Pattern;
				}
				else if (word == "general")
				{
					layout = General;
				}
				else if (word == "symmetric")
				{
					layout = Symmetric;
				}
				else if (word == "skew" || word == "skew-symmetric")
				{
					layout = Skew;
				}
				else if (word == "hermitean" || word == "Hermitean")
				{
					layout = Hermitean;
				}
				else
				{
					cerr << "Unknown Matrix Market header tag '" << word << "'!" << endl;
				}
			}
		}
		else if (line[0] == '%')
		{
			//This is a comment line; we can skip it.
		}
		else
		{
			//We are reading data.
			if (storageType == NoStorage || dataType == NoData || layout == NoLayout)
			{
				cerr << "Incomplete Matrix Market header!" << endl;
				throw exception();
			}

			if (layout == General)
			{
				cerr << "Matrix with unsymmetric nonzero pattern!" << endl;
				throw exception();
			}
			
			if (readDimensions && (nrRows == 0 || nrCols == 0 || nrEntries == 0))
			{
				cerr << "Invalid Matrix Market dimensions!" << endl;
				throw exception();
			}
			
			if (!readDimensions)
			{
				//We still need to read the matrix dimensions.
				sline >> nrRows;
				sline >> nrCols;

				if (layout != General && nrRows != nrCols)
				{
					cerr << "Non-square matrix with symmetry flags!" << endl;
					throw exception();
				}
				
				if (storageType == Coordinate) sline >> nrEntries;
				else nrEntries = nrRows*nrCols;

				if (nrEntries == 0)
				{
					cerr << "No Matrix Market data!" << endl;
					throw exception();
				}
				
				//Allocate arrays.
				nrVertices = nrRows;
				nrVertexWeights = 0;
				nrEdges = 0;
				vertexWeights.clear();
				neighbourRanges.assign(nrVertices, make_int2(0, 0));
				neighbours.clear();
				neighbourWeights.clear();
				edges.clear();
				edges.reserve(nrEntries);
				
				readDimensions = true;
			}
			else
			{
				//Read entry.
				int row, col;
				double re, im;
				
				if (storageType == Coordinate)
				{
					sline >> row;
					sline >> col;
					
					if (row < 1 || row > nrRows || col < 1 || col > nrCols)
					{
						cerr << "Invalid row and/or column index " << row << ", " << col << "!" << endl;
						throw exception();
					}
					
					--row;
					--col;
					
					if (dataType != Pattern)
					{
						sline >> re;
						
						if (dataType == Complex) sline >> im;
						else im = 0.0;
					}
					else
					{
						re = 1.0;
						im = 0.0;
					}
				}
				else
				{
					row = currentRow - 1;
					col = currentCol - 1;
					
					sline >> re;
					
					if (dataType == Complex) sline >> im;
					else im = 0.0;
					
					if (++currentRow >= nrRows)
					{
						currentRow = 1;
						currentCol++;
					}
				}

				//Store edge only if it is below the diagonal.
				if (row > col)
				{
					const float wgt = sqrt(re*re + im*im);

					edges.push_back(Edge(row, col, wgt));
				}
			}
		}
	}

	//Convert edges to neighbour indices.
	neighbours.assign(2*edges.size(), 0);
	neighbourWeights.assign(2*edges.size(), 1.0);
	nrEdges = edges.size();

	//First count the number of neighbours.
	for (vector<Edge>::const_iterator e = edges.begin(); e != edges.end(); ++e)
	{
		neighbourRanges[e->x].y++;
		neighbourRanges[e->y].y++;
	}

	//Sum counts to obtain neighbour offsets.
	for (int i = 1; i < nrVertices; ++i)
	{
		neighbourRanges[i].x = neighbourRanges[i - 1].x + neighbourRanges[i - 1].y;
		neighbourRanges[i - 1].y = neighbourRanges[i - 1].x;
	}

	neighbourRanges[nrVertices - 1].y = neighbourRanges[nrVertices - 1].x;

	//Store actual neighbour indices and weights.
	for (vector<Edge>::const_iterator e = edges.begin(); e != edges.end(); ++e)
	{
		neighbours[neighbourRanges[e->x].y] = e->y;
		neighbourWeights[neighbourRanges[e->x].y] = e->w;
		neighbourRanges[e->x].y++;

		neighbours[neighbourRanges[e->y].y] = e->x;
		neighbourWeights[neighbourRanges[e->y].y] = e->w;
		neighbourRanges[e->y].y++;
	}

	if (nrEdges != (int)edges.size())
	{
		cerr << "Warning: Invalid edge count!" << endl;
		nrEdges = (int)edges.size();
	}

#ifndef NDEBUG
	//Verify storage.
	for (int i = 0; i < nrVertices - 1; ++i)
	{
		if (neighbourRanges[i].y != neighbourRanges[i + 1].x)
		{
			cerr << "Invalid neighbour lists!" << endl;
			throw exception();
		}
	}
#endif

#ifndef NDEBUG
	cerr << "Read a " << nrRows << "x" << nrCols << " matrix from a Matrix Market file." << endl;
#endif
	
	return in;
}

istream &Graph::readMETIS(istream &in)
{
	//Reads a METIS graph file from disk.
	int flags = 0;

	clear();

	//First read header.
	while (in.good())
	{
		string line;

		getline(in, line);

		//Skip all comments and empty lines.
		while (!line.empty() && isspace(line[0])) line.erase(0, 1);

		if (line.length() < 2) continue;
		if (line[0] == '%') continue;

		//Read header information.
		istringstream sline(line);

		if (!(sline >> nrVertices >> nrEdges))
		{
			cerr << "Unable to read METIS header!" << endl;
			throw exception();
		}

		//Read optional arguments.
		nrVertexWeights = 0;

		sline >> flags >> nrVertexWeights;

		break;
	}

	//Determine flags.
	const bool weightedEdges = ((flags % 10) != 0), weightedVertices = (((flags/10) % 10) != 0);
	//bool singleEdges = (((flags/100) % 10) != 0);

	if (nrVertexWeights != 0 && !weightedVertices)
	{
		cerr << "Invalid vertex weight specification!" << endl;
		throw exception();
	}

	if (weightedVertices && nrVertexWeights == 0) nrVertexWeights = 1;

	//Resize graph arrays.
	neighbourRanges.assign(nrVertices, make_int2(0, 0));
	neighbours.reserve(nrEdges);
	neighbourWeights.reserve(nrEdges);
	edges.reserve(nrEdges);

	if (nrVertexWeights > 0) vertexWeights.assign(nrVertexWeights*nrVertices, 0);
	else vertexWeights.clear();

	//Read vertex neighbours from stream.
	bool warnedSelf = false, warnedDuplicates = false;

	for (int i = 0; i < nrVertices; ++i)
	{
		if (!in.good())
		{
			cerr << "Read error!" << endl;
			throw exception();
		}

		string line;

		getline(in, line);

		istringstream sline(line);

		//Read vertex weights.
		if (weightedVertices)
		{
			for (int j = 0; j < nrVertexWeights; ++j) sline >> vertexWeights[nrVertexWeights*i + j];
		}

		//Read vertex neighbours.
		set<int> tmpNeighbours;
		int var;

		neighbourRanges[i].x = neighbours.size();

		while (sline >> var)
		{
			//Verify neighbour index.
			if (var < 1 || var > nrVertices)
			{
				cerr << "Invalid neighbour index!" << endl;
				throw exception();
			}

			//Read edge weight if desired.
			float wgt = 1.0;

			if (weightedEdges) sline >> wgt;

			//Add new neighbour, skipping edges from vertices to themselves.
			//Verify that the neighbour has not occurred earlier.
			if (var - 1 != i)
			{
				if (tmpNeighbours.insert(var).second)
				{
					neighbours.push_back(var - 1);
					neighbourWeights.push_back(wgt);
				}
				else if (!warnedDuplicates)
				{
					warnedDuplicates = true;
					cerr << "Warning, duplicate neighbours!" << endl;
				}
			}
			else if (!warnedSelf)
			{
				warnedSelf = true;
				cerr << "Warning, skipping v--v edges!" << endl;
			}
		}
		
		neighbourRanges[i].y = neighbours.size();

#ifndef NDEBUG
		if (neighbourRanges[i].y > nrVertices + neighbourRanges[i].x)
		{
			cerr << "Too many vertex neighbours!" << endl;
			throw exception();
		}
#endif
	}

	//Create edges.
	for (int i = 0; i < nrVertices; ++i)
	{
		const int2 indices = neighbourRanges[i];

		for (int j = indices.x; j < indices.y; ++j)
		{
			if (i > neighbours[j])
			{
				edges.push_back(Edge(i, neighbours[j], neighbourWeights[j]));
			}
		}
	}

	if (nrEdges != (int)edges.size())
	{
		cerr << "Warning: Invalid edge count!" << endl;
		nrEdges = (int)edges.size();
	}

#ifndef NDEBUG
	cerr << "Read a METIS graph with " << nrVertices << " vertices and " << nrEdges << " edges." << endl;
	//"The graph vertices had " << nrVertexWeights << " weights and the edges " << (weightedEdges ? 1 : 0) << "." << endl;
#endif

	return in;
}

vector<int> Graph::random_shuffle()
{
	//Randomizes the vertex order of the graph.
	vector<int> permutation(nrVertices);
	
	for (int i = 0; i < nrVertices; ++i) permutation[i] = i;

	//Create a random permutation.
	std::random_shuffle(permutation.begin(), permutation.end());
	
	//Permute neighbour ranges.
	if (true)
	{
		vector<int2> tmpRanges(nrVertices);
		
		for (int i = 0; i < nrVertices; ++i) tmpRanges[i] = neighbourRanges[permutation[i]];
		
		neighbourRanges = tmpRanges;
	}
	
	//Create inverse permutation.
	vector<int> invPermutation(nrVertices);
	
	for (int i = 0; i < nrVertices; ++i) invPermutation[permutation[i]] = i;
	
	//Apply inverse permutation to vertex indices and edges.
	for (vector<int>::iterator i = neighbours.begin(); i != neighbours.end(); ++i) *i = invPermutation[*i];

	for (vector<Edge>::iterator i = edges.begin(); i != edges.end(); ++i)
	{
		i->x = invPermutation[i->x];
		i->y = invPermutation[i->y];
	}
	
	return permutation;
}

