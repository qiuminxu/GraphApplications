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
#ifndef CLUSTER_MATCH_H
#define CLUSTER_MATCH_H

#include <vector>

#include "graph.h"

namespace clu
{

#define MATCH_VAL(a, b) (4 + ((a) < (b) ? (a) : (b)))
#define MATCH_MATCHED(a) ((a) >= 4)
#define MATCH_FIRST(a) ((a) - 4)
//We consider a vertex v a satellite if MATCH_SATELLITE*deg(v)*deg(v) <= sum of deg(w) for all neighbours w of v.
#define MATCH_SATELLITE 2L
#define MATCH_BARRIER 0x8000000

class GraphMatching
{
	public:
		GraphMatching();
		virtual ~GraphMatching();
		
		virtual std::vector<int> match(const Graph &) const = 0;
		static bool test(const std::vector<int> &, const Graph &);
};

class GraphMatchingSerial : public GraphMatching
{
	public:
		GraphMatchingSerial();
		~GraphMatchingSerial();
		
		std::vector<int> match(const Graph &) const;
};

}

#endif

