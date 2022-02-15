#pragma once
#include <memory>
#include <string>
#include "chm/distances.hpp"
#include "KnnResults.hpp"

namespace chm {
	template<typename Index>
	void bindIndexImpl(py::module_& m, const std::string& name);

	template<typename Dist>
	DistFunc<Dist> getDistFunc(const SpaceEnum s);

	template<typename ChmAlgo, typename Dist>
	py::tuple knnQueryImpl(std::unique_ptr<ChmAlgo>& algo, const NumpyArray<Dist>& data, const size_t& dim, const size_t& ef, const size_t& K);

	template<typename Index>
	void bindIndexImpl(py::module_& m, const std::string& name) {
		py::class_<Index>(m, name.c_str())
			.def(py::init<const SpaceEnum, const size_t>())
			.def("add_items", &Index::addItems)
			.def("init_index", &Index::init)
			.def("knn_query", &Index::knnQuery)
			.def("set_ef", &Index::setEf);
	}

	template<typename Dist>
	DistFunc<Dist> getDistFunc(const SpaceEnum s) {
		switch(s) {
			case SpaceEnum::EUCLIDEAN:
				return euclideanDistance<Dist>;
			case SpaceEnum::INNER_PRODUCT:
				return innerProductDistance<Dist>;
			default:
				throw std::runtime_error(UNKNOWN_SPACE);
		}
	}

	template<typename ChmAlgo, typename Dist>
	py::tuple knnQueryImpl(std::unique_ptr<ChmAlgo>& algo, const NumpyArray<Dist>& data, const size_t& dim, const size_t& ef, const size_t& K) {
		const auto info = getDataInfo(data, dim);
		KnnResults<Dist> res(info.count, K);

		for(size_t queryIdx = 0; queryIdx < info.count; queryIdx++) {
			auto heap = algo->search(info.ptr + queryIdx * dim, K, ef);

			for(auto neighborIdx = K - 1;; neighborIdx--) {
				{
					const auto& node = heap.top();
					res.setData(queryIdx, neighborIdx, node.dist, node.idx);
				}
				heap.pop();

				if(!neighborIdx)
					break;
			}
		}

		return res.makeTuple();
	}
}
