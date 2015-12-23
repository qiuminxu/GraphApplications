#include <iostream>
#include <sstream>
#include <fstream>
#include <exception>

#include <boost/program_options.hpp>

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

using namespace std;
using namespace boost;

void printStats(istream &in)
{
	//Output the number of edges to stdout.
	while (in.good())
	{
		string line;

		getline(in, line);

		while (!line.empty() && isspace(line[0])) line.erase(0, 1);

		if (line.length() < 2) continue;
		if (line[0] == '%') continue;

		//Print header information.
		istringstream sline(line);
		long a = 0, b = 0, c = 0;

		sline >> a >> b >> c;
		cout << b;
		return;
	}

	cerr << "No header information!" << endl;
	throw std::exception();
}

void readGraph(const string &fileName)
{
	//We are reading a BZIP2 compressed METIS graph.
	ifstream file(fileName.c_str(), ios_base::binary);
	boost::iostreams::filtering_istream inStream;

	inStream.push(boost::iostreams::bzip2_decompressor());
	inStream.push(file);
	
	printStats(inStream);
	file.close();
}

int main(int argc, char **argv)
{
	string prefix = "";
	string fileName = "";

	//Parse command line options.
	try
	{
		program_options::options_description desc("Options");
		
		desc.add_options()
		("help,h", "show this help message")
		("input-file", program_options::value<string>(), "set graph input file")
		("prefix,p", boost::program_options::value<string>(), "set input file prefix");
		
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
	
	//Prepend prefix.
	fileName = prefix + "/" + fileName;
	
	//Make sure the file has the proper extension.
	if (fileName.find(".graph.bz2") == string::npos) fileName += ".graph.bz2";

	cerr << fileName << "..." << endl;

	try
	{
		readGraph(fileName);
	}
	catch (std::exception &e)
	{
		return -1;
	}

	cout << "\t" << fileName << endl;

	return 0;
}

