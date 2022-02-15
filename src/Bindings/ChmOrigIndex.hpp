#pragma once
#include "chm/HnswOrig.hpp"
#include <memory>
#include <string>
#include "pybind.hpp"
#include "SpaceEnum.hpp"

namespace chm {
	template<typename Dist>
	class ChmOrigIndex {
		std::unique_ptr<HnswOrig<Dist>> algo;
		const size_t dim;
		DistFunc<Dist> distFunc;
		size_t ef;

	public:
		void addItems(const NumpyArray<Dist> data);
		ChmOrigIndex(const SpaceEnum spaceEnum, const size_t dim);
		void init(const size_t maxElements, const size_t M, const size_t efConstruction, const unsigned int seed);
		py::tuple knnQuery(const NumpyArray<Dist> data, const size_t K);
		void setEf(const size_t ef);
	};

	template<typename Dist>
	void bindChmOrigIndex(py::module_& m, const std::string& name);

	template<typename Dist>
	inline void ChmOrigIndex<Dist>::addItems(const NumpyArray<Dist> data) {
		const auto info = getDataInfo(data, this->dim);

		for(size_t i = 0; i < info.count; i++)
			this->algo->insert(info.ptr + i * this->dim);
	}

	template<typename Dist>
	inline ChmOrigIndex<Dist>::ChmOrigIndex(const SpaceEnum spaceEnum, const size_t dim) : algo(nullptr), dim(dim), distFunc{}, ef(DEFAULT_EF) {
		switch(spaceEnum) {
			case SpaceEnum::EUCLIDEAN:
				this->distFunc = euclideanDistance<Dist>;
				break;
			case SpaceEnum::INNER_PRODUCT:
				this->distFunc = innerProductDistance<Dist>;
				break;
			default:
				throw std::runtime_error(UNKNOWN_SPACE);
		}
	}

	template<typename Dist>
	inline void ChmOrigIndex<Dist>::init(const size_t maxElements, const size_t M, const size_t efConstruction, const unsigned int seed) {
		this->algo = std::make_unique<HnswOrig<Dist>>(this->dim, this->distFunc, efConstruction, M, maxElements, seed);
	}

	template<typename Dist>
	inline py::tuple ChmOrigIndex<Dist>::knnQuery(const NumpyArray<Dist> data, const size_t K) {
		const auto info = getDataInfo(data, this->dim);
		KnnResults<Dist> res(info.count, K);

		for(size_t queryIdx = 0; queryIdx < info.count; queryIdx++) {
			auto heap = this->algo->search(info.ptr + queryIdx * this->dim, K, this->ef);

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

	template<typename Dist>
	inline void ChmOrigIndex<Dist>::setEf(const size_t ef) {
		this->ef = ef;
	}

	template<typename Dist>
	void bindChmOrigIndex(py::module_& m, const std::string& name) {
		py::class_<ChmOrigIndex<Dist>>(m, name.c_str())
			.def(py::init<const SpaceEnum, const size_t>())
			.def("add_items", &ChmOrigIndex<Dist>::addItems)
			.def("init_index", &ChmOrigIndex<Dist>::init)
			.def("knn_query", &ChmOrigIndex<Dist>::knnQuery)
			.def("set_ef", &ChmOrigIndex<Dist>::setEf);
	}
}
