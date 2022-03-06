#pragma once
#include <functional>

namespace chm {
	template<typename Dist>
	Dist euclideanDistance(const void* nodePtr, const void* queryPtr, const void* dimPtr) {
		const auto dim = *(const size_t* const)(dimPtr);
		const Dist* const node = (const Dist* const)nodePtr;
		const Dist* const query = (const Dist* const)queryPtr;
		Dist res = 0;

		for(size_t i = 0; i < dim; i++) {
			const auto diff = node[i] - query[i];
			res += diff * diff;
		}

		return res;
	}

	template<typename Dist>
	Dist innerProductDistance(const void* nodePtr, const void* queryPtr, const void* dimPtr) {
		const auto dim = *(const size_t* const)(dimPtr);
		const Dist* const node = (const Dist* const)nodePtr;
		const Dist* const query = (const Dist* const)queryPtr;
		Dist res = 0;

		for(size_t i = 0; i < dim; i++)
			res += node[i] * query[i];

		return Dist{1} - res;
	}

	template<typename Dist>
	using DistFunc = std::function<Dist(const void*, const void*, const void*)>;
}
