#pragma once
#include "Unique.hpp"

namespace chm {
	template<typename Coord>
	Coord euclideanDistance(const ConstIter<Coord>& node, const ConstIter<Coord>& query, const size_t dim) {
		Coord res = 0;

		for(size_t i = 0; i < dim; i++) {
			const auto diff = *(node + i) - *(query + i);
			res += diff * diff;
		}

		return res;
	}

	template<typename Coord>
	Coord innerProductDistance(const ConstIter<Coord>& node, const ConstIter<Coord>& query, const size_t dim) {
		Coord res = 0;

		for(size_t i = 0; i < dim; i++)
			res += *(node + i) * *(query + i);

		return Coord{1} - res;
	}
}
