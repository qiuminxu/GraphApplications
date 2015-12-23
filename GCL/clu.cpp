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
#include <exception>
#include <algorithm>
#include <functional>
#include <numeric>

#include <graph.h>
#include <cluster.h>
#include <clustertbb.h>
#include <clustercuda.h>

#include <cuda.h>
#include <tbb/tbb.h>

#include <boost/program_options.hpp>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

using namespace std;
using namespace clu;
using namespace boost;
using namespace tbb;

//Calculate mean and standard deviation.
pair<double, double> getStats(const vector<double> &v)
{
	const double avg = accumulate(v.begin(), v.end(), 0.0)/(double)v.size();
	vector<double> w(v.size());
	
	transform(v.begin(), v.end(), w.begin(), std::bind2nd(std::minus<double>(), avg));
	const double dev = inner_product(w.begin(), w.end(), w.begin(), 0.0)/(double)w.size();
	
	return pair<double, double>(avg, dev);
}

ostream &operator << (ostream &out, const pair<double, double> &rhs)
{
	out << rhs.first << "\t" << rhs.second;
	return out;
}
#if 0
void initCUDA(CUdevice &device, CUcontext &context, const int &deviceIndex)
{
	//Initialise CUDA.
	if (cuInit(0) != CUDA_SUCCESS)
	{
		cerr << "Unable to initialise CUDA!" << endl;
		throw std::exception();
	}
	
	if (cuDeviceGet(&device, deviceIndex) != CUDA_SUCCESS)
	{
		cerr << "Cannot select device " << deviceIndex << "!" << endl;
		throw std::exception();
	}
	
	if (cuCtxCreate(&context, 0, device) != CUDA_SUCCESS)
	{
		cerr << "Unable to create CUDA context!" << endl;
		throw std::exception();
	}

	/*
	if (cudaSetDevice(deviceIndex) != cudaSuccess)
	{
		cerr << "Unable to set device!" << endl;
		throw std::exception();
	}
	
	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, deviceIndex);
	*/
	char deviceName[256];
	
	cuDeviceGetName(deviceName, 256, device);
	//cuCtxSynchronize();
	
	cerr << "Using CUDA device " << deviceIndex << " (" << deviceName << ")." << endl;
}
#endif

void exitCUDA(CUdevice &device, CUcontext &context)
{
	//Destroy context.
	//cuCtxDestroy(context);
}

int main(int argc, char **argv)
{
	string fileName = "", shortFileName = "";
	string gnuplotFileName = "";
	//CUdevice GPUDevice;
	//CUcontext GPUContext;
	//int GPUDeviceIndex = 0;
	int nrTimeAvg = 1;
	int runningMode = 0;
	bool randomiseVertices = false;
	
	//Parse command line options.
	try
	{
		program_options::options_description desc("Options");
		
		desc.add_options()
		("help,h", "show this help message")
		("input-file", program_options::value<string>(), "set graph input file")
		("random,r", "randomise the graph vertex ordering")
//		("device,d", boost::program_options::value<int>(), "set GPU device index")
		("repetitions,a", program_options::value<int>(), "set number of timing repetitions")
		("mode,m", program_options::value<int>(), "set running mode (1 = TBB, 2 = CUDA, 3 = TBB (scaling))");
		
		program_options::positional_options_description pos;
		
		pos.add("input-file", -1);
		
		program_options::variables_map varMap;
		program_options::store(program_options::command_line_parser(argc, argv).options(desc).positional(pos).run(), varMap);
		program_options::notify(varMap);
		
		if (varMap.count("help"))
		{
			cerr << desc << endl;
			return 1;
		}
		
		if (varMap.count("random")) randomiseVertices = true;
		//if (varMap.count("device")) GPUDeviceIndex = max(0, varMap["device"].as<int>());
		if (varMap.count("repetitions")) nrTimeAvg = max(1, varMap["repetitions"].as<int>());
		if (varMap.count("mode")) runningMode = varMap["mode"].as<int>();
		if (varMap.count("input-file")) fileName = varMap["input-file"].as<string>();
		
		if (fileName == "")
		{
			cerr << "You have to specify an input file!" << endl;
			throw std::exception();
		}
	}
	catch (std::exception &e)
	{
		cerr << "Invalid command line arguments!" << endl;
		return -1;
	}
	
	//Initialise cuda.
#if 0
	if (runningMode == 2)
	{
		try
		{
			initCUDA(GPUDevice, GPUContext, GPUDeviceIndex);
		}
		catch (std::exception &e)
		{
			cerr << "Unable to initialise CUDA!" << endl;
			return -1;
		}
	}
#endif

	cerr << "Creating clustering of '" << fileName << "' by averageing " << nrTimeAvg << " times..." << endl;

	//Read graph from disk.
	Graph graph;

	try
	{
		//We are reading a BZIP2 compressed METIS graph.
		ifstream file(fileName.c_str(), ios_base::binary);
		iostreams::filtering_istream inStream;

		inStream.push(iostreams::bzip2_decompressor());
		inStream.push(file);
		graph.readMETIS(inStream);
		file.close();
	
		shortFileName = fileName.substr(1 + fileName.find_last_of("/\\"));
		shortFileName = shortFileName.substr(0, shortFileName.find(".graph.bz2"));
	}
	catch (std::exception &e)
	{
		cerr << "An exception occured when reading " << fileName << " from disk!" << endl;
		return -1;
	}
	
	//Randomise graph if desired.
	srand(12345);
	
	if (randomiseVertices) graph.random_shuffle();
	
	if (runningMode > 0 && runningMode <= 2)
	{
		task_scheduler_init tbbScheduler;
		Cluster *cluster = 0;
		vector<int> component(graph.nrVertices);
		vector<double> times(nrTimeAvg), modularities(nrTimeAvg), nrComponents(nrTimeAvg);
		
		if (runningMode == 1)
		{
			cerr << "CPU parallel TBB mode." << endl;
			cluster = new ClusterTBB();
		}
		else if (runningMode == 2)
		{
			cerr << "GPU parallel CUDA mode." << endl;
			cluster = new ClusterCUDA();
		}
		
		//Perform clustering the desired number of times.
		for (int j = 0; j < nrTimeAvg; ++j)
		{
			//if (runningMode == 2) cuCtxSynchronize();
			tick_count t0 = tick_count::now();
			
			component = cluster->cluster(graph, 1.0, 0);

			//if (runningMode == 2) cuCtxSynchronize();
			tick_count t1 = tick_count::now();
			
			times[j] = (t1 - t0).seconds();
			modularities[j] = Cluster::modularity(graph, component);
			nrComponents[j] = *max_element(component.begin(), component.end());
			
			if (randomiseVertices) graph.random_shuffle();
		}
		
		delete cluster;
		
		cout << shortFileName << "\t" << graph.nrVertices << "\t" << graph.nrEdges << "\t" << scientific << getStats(modularities) << "\t" << getStats(times) << "\t" << getStats(nrComponents) << endl;
	}
	else if (runningMode == 3)
	{
		//CPU parallel with TBB.
		cerr << "CPU parallel TBB mode." << endl;
		
		const int maxNrThreads = task_scheduler_init::default_num_threads();
		
		for (int i = 1; i <= maxNrThreads; ++i)
		{
			task_scheduler_init tbbScheduler(i);
			
			Cluster *cluster = new ClusterTBB();
			vector<int> component;
			
			tick_count t0 = tick_count::now();
			
			for (int j = 0; j < nrTimeAvg; ++j) component = cluster->cluster(graph, 1.0, 0);
			
			tick_count t1 = tick_count::now();
			
			delete cluster;
			
			cout << i << "\t" << Cluster::modularity(graph, component) << "\t" << (t1 - t0).seconds()/(double)nrTimeAvg << endl;
		}
	}
	else
	{
		cerr << "Invalid mode " << runningMode << "!" << endl;
		return -1;
	}
	
	//Shut down CUDA.
#if 0
	if (runningMode == 2)
	{
		exitCUDA(GPUDevice, GPUContext);
	}
#endif
	
	return 0;
}

