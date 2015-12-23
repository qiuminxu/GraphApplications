#include <iostream>
#include <sstream>
#include <fstream>
#include <exception>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

void printStats(std::istream &in, const bool &isMatrix)
{
	//Output the number of edges to stdout.
	while (in.good())
	{
		std::string line;

		std::getline(in, line);

		while (!line.empty() && isspace(line[0])) line.erase(0, 1);

		if (line.length() < 2) continue;
		if (line[0] == '%') continue;

		//Print header information.
		std::istringstream sline(line);
		long a = 0, b = 0, c = 0;

		sline >> a >> b >> c;

		if (isMatrix) std::cout << c;
		else std::cout << b;

		return;
	}

	std::cerr << "No header information!" << std::endl;
	throw std::exception();
}

void readGraph(const std::string &fileName)
{
	if (fileName.find(".graph.bz2") != std::string::npos)
	{
		//We are reading a BZIP2 compressed METIS graph.
		std::ifstream file(fileName.c_str(), std::ios_base::binary);
		boost::iostreams::filtering_istream inStream;

		inStream.push(boost::iostreams::bzip2_decompressor());
		inStream.push(file);
		
		printStats(inStream, false);
		file.close();
	}
	else if (fileName.find(".mtx.bz2") != std::string::npos)
	{
		//We are reading a BZIP2 compressed Matrix Market file.
		std::ifstream file(fileName.c_str(), std::ios_base::binary);
		boost::iostreams::filtering_istream inStream;

		inStream.push(boost::iostreams::bzip2_decompressor());
		inStream.push(file);
		
		printStats(inStream, true);
		file.close();
	}
	else if (fileName.find(".graph") != std::string::npos)
	{
		//We are reading a METIS graph.
		std::ifstream file(fileName.c_str());

		printStats(file, false);
		file.close();
	}
	else if (fileName.find(".mtx") != std::string::npos)
	{
		//We are reading a Matrix Market file.
		std::ifstream file(fileName.c_str());

		printStats(file, true);
		file.close();
	}
	else
	{
		std::cerr << "Unknown filename extension for '" << fileName << "'!" << std::endl;
		throw std::exception();
	}
}

int main(int argc, char **argv)
{
	if (argc != 2)
	{
		std::cerr << "Usage: " << argv[0] << " foo.graph" << std::endl;
		return -1;
	}

	const std::string fileName = argv[1];

	std::cerr << fileName << "..." << std::endl;

	try
	{
		readGraph(fileName);
	}
	catch (std::exception &e)
	{
		return -1;
	}

	std::cout << "\t" << fileName << std::endl;

	return 0;
}

