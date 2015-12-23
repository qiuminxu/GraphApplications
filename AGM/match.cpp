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
#include <iomanip>
#include <fstream>
#include <vector>
#include <set>
#include <exception>
#include <climits>
#include <cmath>

#include <cuda.h>

#include <graph.h>
#include <matchcpu.h>
#include <matchgpu.h>

#include <boost/program_options.hpp>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

using namespace std;
using namespace mtc;

/*void initCUDA(CUdevice &device, int &nrThreads, const int &deviceIndex, const int &nrVertices)
{
	//Initialise CUDA.
	//if (cuInit(0) != CUDA_SUCCESS)
	//{
	//	cerr << "Unable to initialise CUDA!" << endl;
	//	throw exception();
	//}

	int nrDevices = 0;

	cudaGetDeviceCount(&nrDevices);

	if (deviceIndex >= nrDevices || deviceIndex < 0)
	{
		cerr << "Invalid GPU device index (" << deviceIndex << ")!" << endl;
		throw exception();
	}

	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, deviceIndex);
	
	//Adjust the number of threads per block if we have a large number of vertices.
	if (nrThreads*prop.maxGridSize[0] <= 2*nrVertices)
	{
		nrThreads = prop.maxThreadsPerBlock;
	}

	if (cuDeviceGet(&device, deviceIndex) != CUDA_SUCCESS)
	{
		cerr << "Unable to retrieve device!" << endl;
		throw exception();
	}

	cerr << "Using CUDA device " << deviceIndex << " (" << prop.name << "), with " << nrThreads << " threads per block." << endl;
}*/

string readGraph(Graph &graph, const string &fileName)
{
	string shortFileName = fileName.substr(1 + fileName.find_last_of("/\\"));
	
	if (fileName.find(".graph.bz2") != string::npos)
	{
		//We are reading a BZIP2 compressed METIS graph.
		ifstream file(fileName.c_str(), ios_base::binary);
		boost::iostreams::filtering_istream inStream;

		inStream.push(boost::iostreams::bzip2_decompressor());
		inStream.push(file);
		graph.readMETIS(inStream);
		file.close();
	
		shortFileName = shortFileName.substr(0, shortFileName.find(".bz2"));
	}
	else if (fileName.find(".mtx.bz2") != string::npos)
	{
		//We are reading a BZIP2 compressed Matrix Market file.
		ifstream file(fileName.c_str(), ios_base::binary);
		boost::iostreams::filtering_istream inStream;

		inStream.push(boost::iostreams::bzip2_decompressor());
		inStream.push(file);
		graph.readMatrixMarket(inStream);
		file.close();
		
		shortFileName = shortFileName.substr(0, shortFileName.find(".bz2"));
	}
	else if (fileName.find(".graph") != string::npos)
	{
		//We are reading a METIS graph.
		ifstream file(fileName.c_str());

		graph.readMETIS(file);
		file.close();
	}
	else if (fileName.find(".mtx") != string::npos)
	{
		//We are reading a Matrix Market file.
		ifstream file(fileName.c_str());

		graph.readMatrixMarket(file);
		file.close();
	}
	else
	{
		cerr << "Unknown filename extension for '" << fileName << "'!" << endl;
		throw exception();
	}

	return shortFileName;
}

GraphMatching *getMatcher(const Graph &graph, const int &type, const int &nrThreads, const unsigned int &barrier)
{
	//Returns a matcher of the desired type.
	if (type ==       0) return new GraphMatchingCPURandom(graph);
	else if (type ==  1) return new GraphMatchingCPUStatMinDeg(graph);
	else if (type ==  2) return new GraphMatchingCPUKarpSipser(graph);
	else if (type ==  3) return new GraphMatchingCPUMinDeg(graph);
	else if (type ==  4) return new GraphMatchingCPUWeighted(graph);
	else if (type ==  5) return new GraphMatchingCPUWeightedEdge(graph);
	else if (type ==  6) return new GraphMatchingGPURandom(graph, nrThreads, barrier);
	else if (type ==  7) return new GraphMatchingGPURandomMaximal(graph, nrThreads, barrier);
	else if (type ==  8) return new GraphMatchingGPUWeighted(graph, nrThreads, barrier);
	else if (type ==  9) return new GraphMatchingGPUWeightedMaximal(graph, nrThreads, barrier);
	else
	{
		cerr << "Unknown matching type!" << endl;
		throw exception();
	}

	return 0;
}

void getStats(double &avg, double &dev, const vector<double> &data)
{
	//Calculates the average and standard deviation of the recorded data.
	double sum = 0.0;
	
	for (vector<double>::const_iterator i = data.begin(); i != data.end(); ++i) sum += *i;
	
	avg = sum/(double)data.size();
	sum = 0.0;

	for (vector<double>::const_iterator i = data.begin(); i != data.end(); ++i) sum += (*i - avg)*(*i - avg);
	
	dev = sqrt(sum/(double)data.size());
}

int main(int argc, char **argv)
{
	string fileName = "", shortFileName = "";
	string gnuplotFileName = "";
	set<int> matchTypes;
	set<int> CPUNrThreads;
	int nrTimeAvg = 1;
	//This should amount to choosing p = 0.534059... .
	unsigned int barrier = 0x88B81733;
	
	int GPUDeviceIndex = 0;
	int GPUNrThreadsPerBlock = 256;
	CUdevice GPUDevice;

	bool outputScaleData = false;
	bool performTest = false;
	bool randomiseVertices = false;
	
	matchTypes.insert(0);

	//Parse command line options.
	try
	{
		boost::program_options::options_description desc("Options");
		
		desc.add_options()
		("help,h", "show this help message")
		("input-file", boost::program_options::value<string>(), "set graph input file")
		("test,t", "verify all generated matchings")
		("random,r", "randomise the graph vertex ordering")
		("device,d", boost::program_options::value<int>(), "set GPU device index")
		("repetitions,a", boost::program_options::value<int>(), "set number of timing repetitions")
		("barrier,b", boost::program_options::value<int>(), "set selection barrier (between 0 and 16)")
		("gnuplot", boost::program_options::value<string>(), "output gnuplot data to file")
		("scaledata", "output thread scaling data")
		("threads", boost::program_options::value<string>(), "number of TBB threads")
		("match,m", boost::program_options::value<string>(), 
		 "desired matching types:\n"
		 "0  = CPU random,\n"
		 "1  = CPU static minimum degree,\n"
		 "2  = CPU Karp--Sipser,\n"
		 "3  = CPU dynamic minimum degree,\n"
		 "4  = CPU weighted,\n"
		 "5  = CPU weighted (edge-based),\n"
		 "6  = GPU random,\n"
		 "7  = GPU maximal random,\n"
		 "8  = GPU weighted,\n"
		 "9  = GPU maximal weighted,\n"
		 "10 = TBB random,\n"
		 "11 = TBB weighted.");
		
		boost::program_options::positional_options_description pos;
		
		pos.add("input-file", -1);
		
		boost::program_options::variables_map varMap;
		boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(pos).run(), varMap);
		boost::program_options::notify(varMap);
		
		if (varMap.count("help"))
		{
			cerr << desc << endl;
			return 1;
		}
		
		if (varMap.count("test")) performTest = true;
		if (varMap.count("random")) randomiseVertices = true;
		if (varMap.count("device")) GPUDeviceIndex = max(0, varMap["device"].as<int>());
		if (varMap.count("repetitions")) nrTimeAvg = max(1, varMap["repetitions"].as<int>());
		if (varMap.count("gnuplot")) gnuplotFileName = varMap["gnuplot"].as<string>();
		if (varMap.count("scaledata")) outputScaleData = true;
		
		if (varMap.count("barrier"))
		{
			const int b = max(0, varMap["barrier"].as<int>());

                        if (b >= 16) barrier = 0xffffffff;
                        else barrier = (unsigned int)((long)b*0x10000000L);
		}

		if (varMap.count("threads"))
		{
			string types = varMap["threads"].as<string>();
			istringstream stypes(types);
			int t;
			
			CPUNrThreads.clear();

			while (stypes >> t) CPUNrThreads.insert(t);
		}

		if (varMap.count("match"))
		{
			string types = varMap["match"].as<string>();
			istringstream stypes(types);
			int t;
			
			matchTypes.clear();

			while (stypes >> t) matchTypes.insert(t);
		}

		if (varMap.count("input-file")) fileName = varMap["input-file"].as<string>();
		
		if (fileName == "")
		{
			cerr << "You have to specify an input file!" << endl;
			throw exception();
		}
	}
	catch (exception &e)
	{
		cerr << "Invalid command line arguments!" << endl;
		return -1;
	}

	cerr << fileName << "..." << endl;

	//Read graph from disk.
	Graph graph;

	try
	{
		shortFileName = readGraph(graph, fileName);
	}
	catch (exception &e)
	{
		cerr << "An exception occured when reading " << fileName << " from disk!" << endl;
		return -1;
	}
	
	//Initialise CUDA.
	/*try
	{
		initCUDA(GPUDevice, GPUNrThreadsPerBlock, GPUDeviceIndex, graph.nrVertices);
	}
	catch (exception &e)
	{
		cerr << "Unable to initialise CUDA!" << endl;
		return -1;
	}*/

        
	GPUDevice = 0;
	cudaSetDevice(GPUDevice);
	GPUNrThreadsPerBlock = 256;
	GPUDeviceIndex = 0;
	
	
	//Open GNUplot file.
	ofstream gnuplotFile;

	if (gnuplotFileName != "")
	{
		gnuplotFile.open(gnuplotFileName.c_str(), ios_base::out | ios_base::app);

		if (!gnuplotFile.good())
		{
			cerr << "Unable to open '" << gnuplotFileName << "' for output!" << endl;
			return -1;
		}
	
		if (!outputScaleData)
		{
			//Output graph file.
			gnuplotFile << shortFileName << "\t" << graph.nrVertices << "\t" << graph.nrEdges;
			
			//Output minimum and maximum graph degrees.
			int minDeg = INT_MAX, maxDeg = INT_MIN;
			
			for (vector<int2>::const_iterator i = graph.neighbourRanges.begin(); i != graph.neighbourRanges.end(); ++i)
			{
				const int deg = i->y - i->x;
				
				if (minDeg > deg) minDeg = deg;
				if (maxDeg < deg) maxDeg = deg;
			}
			
			gnuplotFile << "\t" << minDeg << "\t" << maxDeg;
		}
	}
	
#ifndef MATCH_INTERMEDIATE_COUNT
	cout << shortFileName << " " << graph.nrVertices << " " << graph.nrEdges << " " << barrier << endl;
#else
	cerr << "Warning: Running in INTERMEDIATE mode!" << endl;

	nrTimeAvg = 1;
#endif

	//Perform all desired greedy matchings.
	//for (set<int>::const_iterator i = matchTypes.begin(); i != matchTypes.end(); ++i)
	set<int>::const_iterator i = matchTypes.begin(); 
	{
	//	for (set<int>::const_iterator j = CPUNrThreads.begin(); j != CPUNrThreads.end(); ++j)
		set<int>::const_iterator j = CPUNrThreads.begin();
		{
			
			//Store data for all iterations.
			vector<double> matchingSizes(nrTimeAvg, 0);
			vector<double> matchingWeights(nrTimeAvg, 0.0);
			vector<double> totalTimes(nrTimeAvg, 0.0);
			vector<double> matchTimes(nrTimeAvg, 0.0);
			
			//Ensure we use the same graph permutations.
			Graph graph2 = graph;

			srand(12345);

			//Average the time over the desired number of iterations.
			for (int k = 0; k < nrTimeAvg; ++k)
			{
				//Randomise the graph if desired.
				if (randomiseVertices) graph2.random_shuffle();

				
				//FIXME: It really bugs me that I cannot calculate averages without putting this in the inner loop.
				/*CUcontext GPUContext;

				if (cuCtxCreate(&GPUContext, 0, GPUDevice) != CUDA_SUCCESS)
				{
					cerr << "Unable to create CUDA context!" << endl;
					return -1;
				}*/
				
				
				//Initialise timers.
				cudaEvent_t t0, t1, t2, t3;

				cudaEventCreate(&t0);
				cudaEventCreate(&t1);
				cudaEventCreate(&t2);
				cudaEventCreate(&t3);
				
				cudaEventRecord(t0, 0);
				cudaEventSynchronize(t0);

				//Generate matching of the desired type.
				vector<int> match;

				try
				{
					GraphMatching *matcher = getMatcher(graph2, *i, GPUNrThreadsPerBlock, barrier);
					
					match = matcher->initialMatching();
					matcher->performMatching(match, t1, t2);
				
					delete matcher;
				}
				catch (exception &e)
				{
					cerr << "An exception occured when matching!" << endl;
					return -1;
				}
			
				cudaEventRecord(t3, 0);
				cudaEventSynchronize(t3);
				
				//Measure the total elapsed time (including data transfer) and the calculation time.
				float time0, time1;
				
				cudaEventElapsedTime(&time0, t0, t3);
				cudaEventElapsedTime(&time1, t1, t2);

				//Destroy timers.
				cudaEventDestroy(t3);
				cudaEventDestroy(t2);
				cudaEventDestroy(t1);
				cudaEventDestroy(t0);
		
				/*
				//Destroy CUDA context.
				cuCtxDestroy(GPUContext);
				*/
				
				//Test matching if desired.
				if (performTest)
				{
					if (!GraphMatching::testMatching(match, graph2))
					{
						cerr << "Invalid matching!" << endl;
						return -1;
					}
				}
				
				//Determine matching weight and size.
				double matchingWeight = 0.0;
				long matchingSize = 0;
				
				GraphMatching::getWeight(matchingWeight, matchingSize, match, graph2);

				//Store benchmark data.
				matchingSizes[k] = matchingSize;
				matchingWeights[k] = matchingWeight;
				totalTimes[k] = time0;
				matchTimes[k] = time1;
			}

#ifndef MATCH_INTERMEDIATE_COUNT
			double avg = 0.0, dev = 0.0;
			
			cout << *i << " ";
			if (!outputScaleData) gnuplotFile << "\t";
			
				cout << GPUNrThreadsPerBlock << " ";
				gnuplotFile << GPUNrThreadsPerBlock << "\t";
			
			//Output averages and standard deviations of matching size, weight, and time.
			getStats(avg, dev, matchingSizes);
			cout << avg << " " << dev << " ";
			gnuplotFile << avg << "\t" << dev << "\t";
			
			getStats(avg, dev, matchingWeights);
			cout << avg << " " << dev << " ";
			gnuplotFile << avg << "\t" << dev << "\t";
			
			getStats(avg, dev, totalTimes);
			cout << avg << " " << dev << " ";
			gnuplotFile << avg << "\t" << dev << "\t";
			
			getStats(avg, dev, matchTimes);
			cout << avg << " " << dev << " ";
			gnuplotFile << avg << "\t" << dev << "\t";
			
			cout << endl;
			gnuplotFile << (double)barrier/(double)(UINT_MAX);
			if (outputScaleData) gnuplotFile << endl;
			gnuplotFile.flush();
#endif
			
			//If we are not using TBB, then we do not have to select all possible thread configurations.
			//break;
		}
	}

	//Close GNUplot file.
	gnuplotFile << endl;
	gnuplotFile.close();

	return 0;
}

