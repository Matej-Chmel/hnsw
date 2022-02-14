#pragma once
#include <functional>

namespace chm {
	template<typename Dist>
	Dist euclideanDistance(const Dist* const node, const Dist* const query, const size_t dim) {
		Dist res = 0;

		for(size_t i = 0; i < dim; i++) {
			const auto diff = node[i] - query[i];
			res += diff * diff;
		}

		return res;
	}

	template<typename Dist>
	Dist innerProductDistance(const Dist* const node, const Dist* const query, const size_t dim) {
		Dist res = 0;

		for(size_t i = 0; i < dim; i++)
			res += node[i] * query[i];

		return Dist{1} - res;
	}

	template<typename Dist>
	using DistFunc = std::function<Dist(const Dist* const, const Dist* const, const size_t)>;
}
