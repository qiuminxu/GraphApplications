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

#include <graph.h>
#include <vis.h>
#include <cluster.h>
#include <clustertbb.h>
#include <clustercuda.h>

#include <cmath>

#include <SDL.h>

#include <boost/program_options.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/bzip2.hpp>

using namespace clu;
using namespace std;

class pixel
{
	public:
		pixel(const unsigned char &_b, const unsigned char &_g, const unsigned char &_r) :
			r(_r), g(_g), b(_b)
		{

		};
		
		~pixel()
		{

		}
		
		inline pixel operator |= (const pixel &a)
		{
			r |= a.r;
			g |= a.g;
			b |= a.b;
			
			return *this;
		}
		
		unsigned char r, g, b;
};

class DrawerSDL : public Drawer
{
	public:
		DrawerSDL(SDL_Surface *, const int &);
		~DrawerSDL();
		
		void nextSlide();
		void prevSlide();
		
		void drawGraphMatrix(const Graph &);
		void drawGraphMatrixPermuted(const Graph &, const std::vector<int> &);
		void drawGraphMatrixPermutedClustering(const Graph &, const std::vector<int> &, const std::vector<int> &);
		void drawGraphCoordinates(const Graph &);
		void drawGraphClustering(const Graph &, const std::vector<int> &);
		
	private:
		void drawLine(int2, int2, const pixel) const;
		void addSlide();
		
		inline int2 convertCoordinate(const float2 &p) const
		{
			return make_int2((int)(coordRect.x + 1 + (coordRect.w - 2)*p.x), (int)(coordRect.y + 1 + (coordRect.h - 2)*(1.0f - p.y)));
		};
		
		inline pixel fromHue(const int &c) const
		{
			const float h = 123.456f*(float)c + 654.123f;
			
			return pixel((unsigned char)(254.0f*min(max(0.0f, 2.0f*cosf(h)), 1.0f)), (unsigned char)(254.0f*min(max(0.0f, 2.0f*cosf(h + 2.0f*M_PI/3.0f)), 1.0f)), (unsigned char)(254.0f*min(max(0.0f, 2.0f*cosf(h + 4.0f*M_PI/3.0f)), 1.0f)));
		};
		
		SDL_Surface * const screen;
		const int size;
		pixel *buffer;
		const int pitch;
		const unsigned int clearColour;
		
		int curSlide;
		vector<SDL_Surface *> slides;
		
		SDL_Rect origRect, permRect, coordRect;
};

int main(int argc, char **argv)
{
	int drawSize = 400;
	bool randomiseVertices = false;
	
	string fileName = "", shortFileName = "";
	string coordinateFileName = "";
	string gnuplotFileName = "";

	//Parse command line options.
	try
	{
		boost::program_options::options_description desc("Options");
		
		desc.add_options()
		("help,h", "show this help message")
		("coordinate-data,c", boost::program_options::value<string>(), "specify vertex coordinates")
		("input-file", boost::program_options::value<string>(), "set graph input file")
		("random,r", "randomise the graph vertex ordering");
		
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
		
		if (varMap.count("random")) randomiseVertices = true;
		if (varMap.count("coordinate-data")) coordinateFileName = varMap["coordinate-data"].as<string>();
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

	cerr << "Creating clustering of '" << fileName << "'..." << endl;

	//Read graph from disk.
	Graph graph;

	try
	{
		//We are reading a BZIP2 compressed METIS graph.
		ifstream file(fileName.c_str(), ios_base::binary);
		boost::iostreams::filtering_istream inStream;

		inStream.push(boost::iostreams::bzip2_decompressor());
		inStream.push(file);
		graph.readMETIS(inStream);
		file.close();
	
		shortFileName = fileName.substr(1 + fileName.find_last_of("/\\"));
		shortFileName = shortFileName.substr(0, shortFileName.find(".bz2"));
	}
	catch (exception &e)
	{
		cerr << "An exception occured when reading " << fileName << " from disk!" << endl;
		return -1;
	}
	
	//Read coordinate data from file if specified.
	vector<int2> coordinates;
	
	if (coordinateFileName != "")
	{
		try
		{
			//We are reading a BZIP2 compressed coordinate file.
			ifstream file(coordinateFileName.c_str(), ios_base::binary);
			boost::iostreams::filtering_istream inStream;

			inStream.push(boost::iostreams::bzip2_decompressor());
			inStream.push(file);
			graph.readCoordinates(inStream);
			file.close();
		}
		catch (exception &e)
		{
			cerr << "An exception occured when reading " << coordinateFileName << " from disk!" << endl;
			return -1;
		}
	}
	
	//Randomise graph if desired.
	if (randomiseVertices) graph.random_shuffle();
	
	//Start SDL to display the results.
	if (SDL_Init(SDL_INIT_VIDEO) != 0)
	{
		cerr << "Unable to initialise SDL: " << SDL_GetError() << endl;
		return -1;
	}
	
	SDL_Surface *screen = SDL_SetVideoMode(3*drawSize, 2*drawSize, 24, SDL_SWSURFACE);
	
	if (!screen)
	{
		cerr << "Unable to create window: " << SDL_GetError() << endl;
		return -1;
	}
	
	//Set caption and clear screen.
	SDL_WM_SetCaption("Cluster", "Cluster");
	SDL_FillRect(screen, &screen->clip_rect, 0x00000000);
	SDL_Flip(screen);
	
	//Create graph drawer.
	DrawerSDL drawer(screen, drawSize);
	
	drawer.drawGraphMatrix(graph);
	drawer.drawGraphCoordinates(graph);
	
	//Enter main loop.
	bool running = true;
	
	while (running)
	{
		//Take care of incoming events.
		int quality = -1;
		SDL_Event event;
		
		while (SDL_PollEvent(&event))
		{
			//Handle keypresses.
			if (event.type == SDL_KEYDOWN)
			{
				const SDLKey k = event.key.keysym.sym;
				
				if (k == SDLK_ESCAPE)
				{
					running = false;
				}
				else if (k == SDLK_LEFT)
				{
					drawer.prevSlide();
				}
				else if (k == SDLK_RIGHT)
				{
					drawer.nextSlide();
				}
				else if (k >= '1' && k <= '9')
				{
					quality = k - (int)('1');
				}
			}
			
			if (event.type == SDL_QUIT)
			{
				running = false;
			}
		}
		
		if (quality >= 0)
		{
			//Start clustering.
			try
			{
				Cluster *cluster = 0;
				
				if (quality == 1) cluster = new ClusterTBB();
				else if (quality == 2) cluster = new ClusterCUDA();
				
				if (cluster)
				{
					vector<int> component = cluster->cluster(graph, 1.0, &drawer);
					
					cout << "Generated clustering with modularity " << Cluster::modularity(graph, component) << "." << endl;
					
					delete cluster;
				}
			}
			catch (exception &e)
			{
				cout << "Unable to generate clustering!" << endl;
			}
			
			SDL_Flip(screen);
		}
		
		SDL_Delay(25);
	}
	
	SDL_Quit();

	return 0;
}

DrawerSDL::DrawerSDL(SDL_Surface *_screen, const int &_size) :
	Drawer(),
	screen(_screen),
	size(_size),
	pitch(screen->w),
#ifdef SAVE_SLIDES
	clearColour(0xffffffff),
#else
	clearColour(0x00000000),
#endif
	curSlide(0)
{
	assert(screen);
	
	coordRect.x = 0; coordRect.y = 0; coordRect.w = 2*size; coordRect.h = 2*size;
	origRect.x = 2*size; origRect.y = 0; origRect.w = size; origRect.h = size;
	permRect.x = 2*size; permRect.y = size; permRect.w = size; permRect.h = size;
}

DrawerSDL::~DrawerSDL()
{
	//Free all slides.
	for (vector<SDL_Surface *>::iterator i = slides.begin(); i != slides.end(); ++i) SDL_FreeSurface(*i);
}

void DrawerSDL::nextSlide()
{
	if (slides.empty()) return;
	
	curSlide = (curSlide + 1) % slides.size();
	SDL_BlitSurface(slides[curSlide], 0, screen, 0);
	SDL_Flip(screen);
}

void DrawerSDL::prevSlide()
{
	if (slides.empty()) return;
	
	curSlide = (curSlide > 0 ? curSlide - 1 : slides.size() - 1);
	SDL_BlitSurface(slides[curSlide], 0, screen, 0);
	SDL_Flip(screen);
}

void DrawerSDL::addSlide()
{
	//Save the current screen as a slide and (optionally) write it to disk.
	//Make copy of screen.
	SDL_Surface *slide = SDL_DisplayFormat(screen);
	
	assert(slide);
	
	SDL_BlitSurface(screen, 0, slide, 0);
	slides.push_back(slide);
	
#ifdef SAVE_SLIDES
	char outFile[] = "map0000.bmp";
	
	sprintf(outFile, "map%04d.bmp", (int)slides.size());
	SDL_SaveBMP(screen, outFile);
#endif
}

void DrawerSDL::drawGraphCoordinates(const Graph &graph)
{
	assert((int)graph.coordinates.size() == graph.nrVertices);
	
	//Clear part of the screen.
	SDL_FillRect(screen, &coordRect, clearColour);
	
	SDL_LockSurface(screen);
	
	buffer = static_cast<pixel *>(screen->pixels);
	
	//Draw graph data as coordinate lines.
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int2 indices = graph.neighbourRanges[i];
		const int2 p0 = convertCoordinate(graph.coordinates[i]);
		
		for (int j = indices.x; j < indices.y; ++j)
		{
			//Draw line to each neighbour, but avoid drawing the same line twice.
			const int ni = graph.neighbours[j].x;
			
			if (i < ni)
			{
				const int2 p1 = convertCoordinate(graph.coordinates[ni]);

#ifdef SAVE_SLIDES				
				drawLine(p0, p1, pixel(0, 0, 0));
#else
				drawLine(p0, p1, pixel(0, 0, 255));
#endif
			}
		}
	}
	
	for (vector<float2>::const_iterator i = graph.coordinates.begin(); i != graph.coordinates.end(); ++i)
	{
		const int2 p = convertCoordinate(*i);
		
#ifdef SAVE_SLIDES
		buffer[p.x + pitch*p.y] = pixel(0, 0, 0);
#else
		buffer[p.x + pitch*p.y] = pixel(255, 255, 0);
#endif
	}
	
	SDL_UnlockSurface(screen);
	SDL_UpdateRect(screen, coordRect.x, coordRect.y, coordRect.w, coordRect.h);
	
	addSlide();
}

void DrawerSDL::drawGraphClustering(const Graph &graph, const vector<int> &cmp)
{
	assert((int)graph.coordinates.size() == graph.nrVertices);
	assert((int)cmp.size() == graph.nrVertices);
	
	//Clear part of the screen.
	SDL_FillRect(screen, &coordRect, clearColour);
	
	SDL_LockSurface(screen);
	
	buffer = static_cast<pixel *>(screen->pixels);
	
	//Draw graph data as coordinate lines.
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int2 indices = graph.neighbourRanges[i];
		const int2 p0 = convertCoordinate(graph.coordinates[i]);
		const int c0 = cmp[i];
		
		for (int j = indices.x; j < indices.y; ++j)
		{
			//Draw line to each neighbour, but avoid drawing the same line twice.
			const int ni = graph.neighbours[j].x;
			
			if (i < ni)
			{
				const int2 p1 = convertCoordinate(graph.coordinates[ni]);
				const int c1 = cmp[ni];

#ifdef SAVE_SLIDES
				if (c0 == c1) drawLine(p0, p1, fromHue(c0));
#else
				drawLine(p0, p1, (c0 == c1 ? fromHue(c0) : pixel(255, 255, 255)));
#endif
			}
		}
	}

#ifdef SAVE_SLIDES
	//Draw graph data as coordinate lines.
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int2 indices = graph.neighbourRanges[i];
		const int2 p0 = convertCoordinate(graph.coordinates[i]);
		const int c0 = cmp[i];
		
		for (int j = indices.x; j < indices.y; ++j)
		{
			//Draw line to each neighbour, but avoid drawing the same line twice.
			const int ni = graph.neighbours[j].x;
			
			if (i < ni)
			{
				const int2 p1 = convertCoordinate(graph.coordinates[ni]);
				const int c1 = cmp[ni];

				if (c0 != c1) drawLine(p0, p1, pixel(0, 0, 0));
			}
		}
	}
#endif
	
	SDL_UnlockSurface(screen);
	SDL_UpdateRect(screen, coordRect.x, coordRect.y, coordRect.w, coordRect.h);
	
	addSlide();
}

void DrawerSDL::drawGraphMatrix(const Graph &graph)
{
	//Clear part of the screen.
	SDL_FillRect(screen, &origRect, clearColour);
	
	SDL_LockSurface(screen);
	
	buffer = static_cast<pixel *>(screen->pixels);
	
	//No permutation.
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int y = ((long)origRect.h*(long)i)/(long)graph.nrVertices;
		pixel *cp = &buffer[(y + origRect.y)*pitch];
		const int2 r = graph.neighbourRanges[i];
		
		for (int j = r.x; j < r.y; ++j)
		{
			cp[(((long)origRect.w*(long)graph.neighbours[j].x)/(long)graph.nrVertices) + origRect.x] = pixel(0, 0, 255);
		}
	}
	
	SDL_UnlockSurface(screen);
	SDL_UpdateRect(screen, origRect.x, origRect.y, origRect.w, origRect.h);
}

void DrawerSDL::drawGraphMatrixPermuted(const Graph &graph, const vector<int> &pi)
{
	//Clear part of the screen.
	SDL_FillRect(screen, &permRect, clearColour);
	
	SDL_LockSurface(screen);
	
	buffer = static_cast<pixel *>(screen->pixels);
	
	//Permuted.
	vector<int> piInv(graph.nrVertices);
	
	for (int i = 0; i < graph.nrVertices; ++i) piInv[pi[i]] = i;
		
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int y = ((long)permRect.h*(long)piInv[i])/(long)graph.nrVertices;
		pixel *cp = &buffer[(y + permRect.y)*pitch];
		const int2 r = graph.neighbourRanges[i];
		
		for (int j = r.x; j < r.y; ++j)
		{
			cp[(((long)permRect.w*(long)piInv[graph.neighbours[j].x])/(long)graph.nrVertices) + permRect.x] = pixel(0, 0, 255);
		}
	}
	
	SDL_UnlockSurface(screen);
	SDL_UpdateRect(screen, permRect.x, permRect.y, permRect.w, permRect.h);
}

void DrawerSDL::drawGraphMatrixPermutedClustering(const Graph &graph, const vector<int> &pi, const vector<int> &cmp)
{
	assert((int)cmp.size() == graph.nrVertices);
	
	//Clear part of the screen.
	SDL_FillRect(screen, &permRect, clearColour);
	
	SDL_LockSurface(screen);
	
	buffer = static_cast<pixel *>(screen->pixels);
	
	//Permuted.
	vector<int> piInv(graph.nrVertices);
	
	for (int i = 0; i < graph.nrVertices; ++i) piInv[pi[i]] = i;
		
	for (int i = 0; i < graph.nrVertices; ++i)
	{
		const int y = ((long)permRect.h*(long)piInv[i])/(long)graph.nrVertices;
		pixel *cp = &buffer[(y + permRect.y)*pitch];
		const int2 r = graph.neighbourRanges[i];
		const int c0 = cmp[i];
		
		for (int j = r.x; j < r.y; ++j)
		{
			const int ni = graph.neighbours[j].x;
			
			cp[(((long)permRect.w*(long)piInv[ni])/(long)graph.nrVertices) + permRect.x] = (c0 == cmp[ni] ? fromHue(c0) : pixel(255, 255, 255));
		}
	}
	
	SDL_UnlockSurface(screen);
	SDL_UpdateRect(screen, permRect.x, permRect.y, permRect.w, permRect.h);
}

void DrawerSDL::drawLine(int2 p0, int2 p1, const pixel c) const
{
	assert(p0.x >= 0 && p0.y >= 0 && p0.x < screen->w && p0.y < screen->h);
	assert(p1.x >= 0 && p1.y >= 0 && p1.x < screen->w && p1.y < screen->h);
	
	//From chapter 36 of Michael Abrash's Black Book of Graphics Programming.
	int w, h, dx;
	pixel *cp;
	
	if ((h = p1.y - p0.y) < 0)
	{
		swap(p0, p1);
		h = -h;
	}
	
	if ((w = p1.x - p0.x) < 0)
	{
		dx = -1;
		w = -w;
	}
	else
	{
		dx = 1;
	}
	
	cp = &buffer[pitch*p0.y + p0.x];
	
	if (w == 0)
	{
		for (int i = h; i-- > 0; cp += pitch) *cp = c;
	}
	else if (h == 0)
	{
		for (int i = w; i-- > 0; cp += dx) *cp = c;
	}
	else if (w == h)
	{
		for (int i = h; i-- > 0; cp += pitch + dx) *cp = c;
	}
	else if (w >= h)
	{
		const int step = w/h, up = (w % h) << 1, down = h << 1;
		int error = (w % h) - (h << 1);
		
		if (step & 1) error += h;
		
		//Initial run.
		for (int i = 1 + (step >> 1); i-- > 0; cp += dx) *cp = c;
		
		cp += pitch;
		
		for (int i = h - 1; i-- > 0; cp += pitch)
		{
			if ((error += up) > 0)
			{
				error -= down;
				
				for (int j = step + 1; j-- > 0; cp += dx) *cp = c;
			}
			else
			{
				for (int j = step; j-- > 0; cp += dx) *cp = c;
			}
		}
		
		//Final run.
		for (int i = (step >> 1) + (up == 0 && !(step & 1) ? 1 : 0); i-- > 0; cp += dx) *cp = c;
	}
	else
	{
		const int step = h/w, up = (h % w) << 1, down = w << 1;
		int error = (h % w) - (w << 1);
		
		if (step & 1) error += w;
		
		//Initial run.
		for (int i = 1 + (step >> 1); i-- > 0; cp += pitch) *cp = c;
		
		cp += dx;
		
		for (int i = w - 1; i-- > 0; cp += dx)
		{
			if ((error += up) > 0)
			{
				error -= down;
				
				for (int j = step + 1; j-- > 0; cp += pitch) *cp = c;
			}
			else
			{
				for (int j = step; j-- > 0; cp += pitch) *cp = c;
			}
		}
		
		//Final run.
		for (int i = (step >> 1) + (up == 0 && !(step & 1) ? 1 : 0); i-- > 0; cp += pitch) *cp = c;
	}
}

