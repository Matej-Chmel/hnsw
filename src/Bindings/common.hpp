#pragma once
#include <cmath>
#include <memory>
#include <string>
#include "KnnResults.hpp"

namespace chm {
	template<typename Index>
	void bindIndexImpl(py::module_& m, const std::string& name);

	template<typename ChmAlgo, typename Dist>
	py::tuple knnQueryImpl(std::unique_ptr<ChmAlgo>& algo, const NumpyArray<Dist>& data, const size_t& dim, const size_t& ef, const size_t& K);

	template<typename Index>
	void bindIndexImpl(py::module_& m, const std::string& name) {
		py::class_<Index>(m, name.c_str())
			.def(py::init<const SpaceEnum, const size_t>())
			.def("add_items", &Index::addItems)
			.def("init_index", &Index::init, py::arg("max_elements"), py::arg("M") = 16, py::arg("ef_construction") = 200, py::arg("seed") = 100)
			.def("knn_query", &Index::knnQuery)
			.def("set_ef", &Index::setEf);
	}

	template<typename Dist>
	void normalizeData(const Dist* const data, std::vector<Dist>& res) {
		const auto dim = res.size();
		Dist norm = 0;

		for(size_t i = 0; i < dim; i++)
			norm += data[i] * data[i];

		norm = Dist(1) / std::sqrt(norm);

		for(size_t i = 0; i < dim; i++)
			res[i] = data[i] * norm;
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
