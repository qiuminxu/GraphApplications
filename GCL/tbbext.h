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
#ifndef TBB_EXTENSIONS_H
#define TBB_EXTENSIONS_H

#include <iterator>
#include <tbb/tbb.h>

/*
This file contains extensions to Intel's Threading Building Blocks library to make it compatible with NVIDIA's Thrust library.

Created by Bas Fagginger Auer.
*/

namespace tbb
{

//Fill an array with a sequence of consecutive numbers in parallel.
template<typename T>
class parallel_sequence_for
{
	public:
		parallel_sequence_for(const T &_v) : v(_v) {};
		~parallel_sequence_for() {};
		
		void operator () (const tbb::blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i) v[i] = i;
		};
		
	private:
		const T v;
};

//Fill a given array with n elements with 0, 1, 2, ..., n - 1 in parallel.
template<typename T>
void parallel_sequence(const T &begin, const T &end)
{
	parallel_sequence_for<T> tmp(begin);
	
	tbb::parallel_for(tbb::blocked_range<size_t>(0, end - begin), tmp);
}

//Sort a list of numbers by their keys.
template<typename T, typename S>
class compare_by_key
{
	public:
		compare_by_key(const T &_v) : v(_v) {};
		~compare_by_key() {};
		
		bool operator () (const S &a, const S &b) const
		{
			return v[a] < v[b];
		};
		
	private:
		const T v;
};

template<typename T, typename S>
void parallel_sort_by_key(const T &begin, const T &end, const S &vbegin)
{
	typedef typename std::iterator_traits<S>::value_type KeyType;
	compare_by_key<T, KeyType> tmp(begin);
	
	//Sort both the indices and the keys.
	tbb::parallel_sort(vbegin, vbegin + (end - begin), tmp);
	tbb::parallel_sort(begin, end);
}

//Fill array with difference between subsequent elements.
template<typename T, typename S, typename BinaryFunction>
class parallel_adjacent_difference_for
{
	public:
		parallel_adjacent_difference_for(const T &_v, const S &_w, const BinaryFunction &_op) : v(_v), w(_w), op(_op) {};
		~parallel_adjacent_difference_for() {};
		
		void operator () (const tbb::blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i) w[i] = op(v[i], v[i - 1]);
		};
		
	private:
		const T v;
		const S w;
		const BinaryFunction &op;
};

template<typename T, typename S, typename BinaryFunction>
void parallel_adjacent_difference(const T &begin, const T &end, const S &out, const BinaryFunction &op)
{
	parallel_adjacent_difference_for<T, S, BinaryFunction> tmp(begin, out, op);
	
	*out = *begin;
	parallel_for(tbb::blocked_range<size_t>(1, end - begin), tmp);
}

//Compress an array by only copying indices of elements that satisfy a certain property.
template<typename T, typename S, typename Predicate>
class parallel_copy_if_for
{
	public:
		parallel_copy_if_for(const T &_v, tbb::concurrent_vector<S> &_w, const Predicate &_p) : v(_v), w(_w), p(_p) {};
		~parallel_copy_if_for() {};
		
		void operator () (const tbb::blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i) if (p(v[i])) w.push_back(i);
		};
		
	private:
		const T v;
		tbb::concurrent_vector<S> &w;
		const Predicate &p;
};

template<typename T, typename S, typename Predicate>
S parallel_copy_if(const T &begin, const T &end, const S &out, const Predicate &pred)
{
	typedef typename std::iterator_traits<S>::value_type KeyType;
	tbb::concurrent_vector<KeyType> copies;
	parallel_copy_if_for<T, KeyType, Predicate> tmp(begin, copies, pred);
	
	parallel_for(tbb::blocked_range<size_t>(0, end - begin), tmp);
	copies.shrink_to_fit();
	parallel_sort(copies.begin(), copies.end());
	std::copy(copies.begin(), copies.end(), out);
	
	return out + copies.size();
}

//Inclusive scan of an array.
template <typename T, typename S>
class parallel_inclusive_scan_red
{
	public:
		parallel_inclusive_scan_red(const T &_v, const T &_w) : v(_v), w(_w), sum(0) {}; 
		parallel_inclusive_scan_red(const parallel_inclusive_scan_red &a, tbb::split) : v(a.v), w(a.w), sum(0) {};
		~parallel_inclusive_scan_red() {};
		
		void operator () (const tbb::blocked_range<size_t> &r, tbb::pre_scan_tag)
		{
			S tmp = sum;
			
			for (size_t i = r.begin(); i != r.end(); ++i) tmp += v[i];
			
			sum = tmp;
		};
		
		void operator () (const tbb::blocked_range<size_t> &r, tbb::final_scan_tag)
		{
			S tmp = sum;
			
			for (size_t i = r.begin(); i != r.end(); ++i) w[i] = (tmp += v[i]);
			
			sum = tmp;
		};
		
		void reverse_join(const parallel_inclusive_scan_red &a)
		{
			sum += a.sum;
		};
		
		void assign(const parallel_inclusive_scan_red &a)
		{
			sum = a.sum;
		};
		
	private:
		const T v, w;
		S sum;
};

template<typename T>
void parallel_inclusive_scan(const T &begin, const T &end, const T &out)
{
	typedef typename std::iterator_traits<T>::value_type KeyType;
	parallel_inclusive_scan_red<T, KeyType> tmp(begin, out);
	
	tbb::parallel_scan(blocked_range<size_t>(0, end - begin), tmp);
}

//Scatter operation.
template<typename T, typename S, typename R>
class parallel_scatter_for
{
	public:
		parallel_scatter_for(const T &_v, const S &_w, const R &_out) : v(_v), w(_w), out(_out) {};
		~parallel_scatter_for() {};
		
		void operator () (const tbb::blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i) out[w[i]] = v[i];
		};
		
	private:
		T v;
		S w;
		R out;
};

template<typename T, typename S, typename R>
void parallel_scatter(const T &begin, const T &end, const S &map, const R &out)
{
	parallel_scatter_for<T, S, R> tmp(begin, map, out);
	
	tbb::parallel_for(blocked_range<size_t>(0, end - begin), tmp);
}

//Gather operation.
template<typename T, typename S, typename R>
class parallel_gather_for
{
	public:
		parallel_gather_for(const T &_v, const S &_w, const R &_out) : v(_v), w(_w), out(_out) {};
		~parallel_gather_for() {};
		
		void operator () (const tbb::blocked_range<size_t> &r) const
		{
			for (size_t i = r.begin(); i != r.end(); ++i)
			{
				out[i] = w[v[i]];
			}
		};
		
	private:
		T v;
		S w;
		R out;
};

template<typename T, typename S, typename R>
void parallel_gather(const T &begin, const T &end, const S &map, const R &out)
{
	parallel_gather_for<T, S, R> tmp(begin, map, out);
	
	tbb::parallel_for(blocked_range<size_t>(0, end - begin), tmp);
}

}

template<typename T, typename S>
class parallel_reduce_red
{
	public:
		parallel_reduce_red(const T &_v) : v(_v), sum(0) {};
		parallel_reduce_red(const parallel_reduce_red &a, tbb::split) : v(a.v), sum(0) {};
		~parallel_reduce_red() {};
		
		void join(const parallel_reduce_red &a)
		{
			sum += a.sum;
		};
		
		void operator () (const tbb::blocked_range<size_t> &r)
		{
			S tmp = 0;
			
			for (size_t i = r.begin(); i != r.end(); ++i) tmp += v[i];
			
			sum += tmp;
		};
		
	private:
		T v;
		
	public:
		S sum;
};

template<typename T>
typename std::iterator_traits<T>::value_type parallel_reduce(const T &begin, const T &end)
{
	typedef typename std::iterator_traits<T>::value_type KeyType;
	parallel_reduce_red<T, KeyType> tmp(begin);
	
	tbb::parallel_reduce(tbb::blocked_range<size_t>(0, end - begin), tmp);
	
	return tmp.sum;
}


#endif
