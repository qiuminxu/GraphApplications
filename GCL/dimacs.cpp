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
/*
DIMACS challenge test program.

Usage:
	./dimacs -m -1 -i 100 foo/bar.graph.bz2

To generate MO#0000#100#foo#bar.ptn and .eval.
*/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <sstream>
#include <exception>

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

	char deviceName[256];
	
	cuDeviceGetName(deviceName, 256, device);
	cuCtxSynchronize();
	
	cerr << "Using CUDA device " << deviceIndex << " (" << deviceName << ")." << endl;
}
#endif
#if 0
void exitCUDA(CUdevice &device, CUcontext &context)
{
	//Destroy context.
	cuCtxDestroy(context);
}
#endif

//void writeDimacs(const string &fileName, const double &modularity, const vector<int> &component, const double &duration, const int &id, const int &GPUDeviceIndex, const CUdevice &GPUDevice, const string &prefix)
void writeDimacs(const string &fileName, const double &modularity, const vector<int> &component, const double &duration, const int &id, const int &GPUDeviceIndex, const string &prefix)
{
	//Find relevant locations in filename.
	const size_t extStart = fileName.rfind(".graph.bz2");
	const size_t pathEnd = fileName.find_last_of("/\\");
	
	if (pathEnd == string::npos)
	{
		cerr << "No path available!" << endl;
		throw std::exception();
	}
	
	const size_t pathStart = fileName.find_last_of("/\\", pathEnd - 1);
	
	if (!(pathStart < pathEnd && pathEnd < extStart))
	{
		cerr << "Invalid filename!" << endl;
		throw std::exception();
	}
	
	//Extract relevant parts.
	const string shortFileName = fileName.substr(pathEnd + 1, extStart - pathEnd - 1);
	const string shortPath = fileName.substr(pathStart + 1, pathEnd - pathStart - 1);
	
	//Generate output files as required by the challenge.
	ostringstream baseName;
	
	baseName << "MO#0000#" << setfill('0') << setw(3) << id << "#" << shortPath << "#" << shortFileName;
	cerr << baseName.str() << endl;
	
	//Create partition file.
	ofstream ptnFile((prefix + "/" + baseName.str() + ".ptn").c_str());
	
	if (!ptnFile.good())
	{
		cerr << "Unable to open DIMACS partition file!" << endl;
		throw std::exception();
	}
	
	for (vector<int>::const_iterator i = component.begin(); i != component.end(); ++i) ptnFile << *i << endl;
	
	ptnFile.close();
	
	//Create evaluation file.
	ofstream evalFile((prefix + "/" + baseName.str() + ".eval").c_str());
	
	if (!evalFile.good())
	{
		cerr << "Unable to open DIMACS partition file!" << endl;
		throw std::exception();
	}
	
	evalFile << setprecision(12) << modularity << endl;
	evalFile << scientific << setprecision(6) << duration << endl;
	evalFile << "Intel(R) Xeon(R) CPU E5620 @ 2.40GHz" << endl;
	
	if (GPUDeviceIndex < 0)
	{
		evalFile << "8, 0, 2" << endl;
		evalFile << endl;
	}
	else
	{
		evalFile << "8, 1, 2" << endl;
		
		char deviceName[256];
		
		//cuDeviceGetName(deviceName, 256, GPUDevice);
		//cuCtxSynchronize();
		
		//evalFile << deviceName << endl;
	}
	
	evalFile.close();
}

int main(int argc, char **argv)
{
	string fileName = "";
	string prefix = "";
	//CUdevice GPUDevice;
//	CUcontext GPUContext;
	//int GPUDeviceIndex = -1;
	int GPUDeviceIndex = 0;
	int dimacsId = 0;
	
	//Parse command line options.
	try
	{
		program_options::options_description desc("Options");
		
		desc.add_options()
		("help,h", "show this help message")
		("input-file", program_options::value<string>(), "set graph input file")
		("prefix,p", boost::program_options::value<string>(), "set output file prefix")
		("device,d", boost::program_options::value<int>(), "set GPU device index, < 0 for TBB")
		("id,i", boost::program_options::value<int>(), "solver identifier");
		
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
		
		if (varMap.count("input-file")) fileName = varMap["input-file"].as<string>();
		if (varMap.count("prefix")) prefix = varMap["prefix"].as<string>();
//		if (varMap.count("device")) GPUDeviceIndex = varMap["device"].as<int>();
		if (varMap.count("id")) dimacsId = varMap["id"].as<int>();
		
		if (fileName == "")
		{
			cerr << "You have to specify an input file!" << endl;
			throw std::exception();
		}
		
		if (dimacsId < 0 || dimacsId > 999)
		{
			cerr << "The DIMACS identifier should lie between 0 and 999!" << endl;
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
	if (GPUDeviceIndex >= 0)
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

	cerr << "Creating clustering of '" << fileName << "'..." << endl;

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
	}
	catch (std::exception &e)
	{
		cerr << "An exception occured when reading " << fileName << " from disk!" << endl;
		return -1;
	}
	
	//Fix random seed for reproducibility.
	srand(12345);
	
	//Start clustering.
	if (true)
	{
		task_scheduler_init tbbScheduler;
		Cluster *cluster = 0;
		vector<int> component(graph.nrVertices);
		
		if (GPUDeviceIndex < 0)
		{
			cerr << "CPU parallel TBB mode." << endl;
			cluster = new ClusterTBB();
		}
		else
		{
			cerr << "GPU parallel CUDA mode." << endl;
			cluster = new ClusterCUDA();
		}
		
		//Perform clustering.
		//if (GPUDeviceIndex >= 0) cuCtxSynchronize();
		const tick_count t0 = tick_count::now();
		
		component = cluster->cluster(graph, 1.0, 0);
		
		//if (GPUDeviceIndex >= 0) cuCtxSynchronize();
		const tick_count t1 = tick_count::now();
		
		delete cluster;
		
		try
		{
			cerr << "Storing DIMACS results..." << endl;
			//writeDimacs(fileName, Cluster::modularity(graph, component), component, (t1 - t0).seconds(), dimacsId, GPUDeviceIndex, GPUDevice, prefix);
			writeDimacs(fileName, Cluster::modularity(graph, component), component, (t1 - t0).seconds(), dimacsId, GPUDeviceIndex, prefix);
			cerr << "Done." << endl;
		}
		catch (std::exception &e)
		{
			cerr << "An exception occured when storing the DIMACS results!" << endl;
			return -1;
		}
		
	}
	
	//Shut down CUDA.
#if 0
	if (GPUDeviceIndex >= 0)
	{
		exitCUDA(GPUDevice, GPUContext);
	}
#endif
	
	return 0;
}

