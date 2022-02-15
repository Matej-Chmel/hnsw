#pragma once
#include <hnswlib/hnswlib.h>
#include <hnswlib/templateSpaces.hpp>
#include <memory>
#include <string>
#include "pybind.hpp"
#include "SpaceEnum.hpp"

namespace chm {
	template<typename Dist>
	class HnswlibIndex {
		std::unique_ptr<hnswlib::HierarchicalNSW<Dist>> algo;
		const size_t dim;
		size_t ef;
		std::unique_ptr<hnswlib::SpaceInterface<Dist>> space;

	public:
		void addItems(const NumpyArray<Dist> data);
		HnswlibIndex(const SpaceEnum spaceEnum, const size_t dim);
		void init(const size_t maxElements, const size_t M, const size_t efConstruction, const unsigned int seed);
		py::tuple knnQuery(const NumpyArray<Dist> data, const size_t K);
		void setEf(const size_t ef);
	};

	template<typename Dist>
	void bindHnswlibIndex(py::module_& m, const std::string& name);

	template<typename Dist>
	inline void HnswlibIndex<Dist>::addItems(const NumpyArray<Dist> data) {
		const auto info = getDataInfo(data, this->dim);

		for(size_t i = 0; i < info.count; i++)
			this->algo->addPoint(info.ptr + i * this->dim, i);
	}

	template<typename Dist>
	inline HnswlibIndex<Dist>::HnswlibIndex(const SpaceEnum spaceEnum, const size_t dim) : algo(nullptr), dim(dim), ef(DEFAULT_EF) {
		switch(spaceEnum) {
			case SpaceEnum::INNER_PRODUCT:
				this->space = std::make_unique<templatedHnswlib::IPSpace<Dist>>(dim);
				break;
			case SpaceEnum::EUCLIDEAN:
				this->space = std::make_unique<templatedHnswlib::EuclideanSpace<Dist>>(dim);
				break;
			default:
				throw std::runtime_error(UNKNOWN_SPACE);
		}
	}

	template<typename Dist>
	inline void HnswlibIndex<Dist>::init(const size_t maxElements, const size_t M, const size_t efConstruction, const unsigned int seed) {
		this->algo = std::make_unique<hnswlib::HierarchicalNSW<Dist>>(this->space.get(), maxElements, M, efConstruction, seed);
	}

	template<typename Dist>
	inline py::tuple HnswlibIndex<Dist>::knnQuery(const NumpyArray<Dist> data, const size_t K) {
		const auto info = getDataInfo(data, this->dim);
		KnnResults<Dist> res(info.count, K);

		for(size_t queryIdx = 0; queryIdx < info.count; queryIdx++) {
			auto queue = this->algo->searchKnn(info.ptr + queryIdx * this->dim, K);

			for(auto neighborIdx = K - 1;; neighborIdx--) {
				const auto& p = queue.top();
				res.setData(queryIdx, neighborIdx, p.first, p.second);
				queue.pop();

				if(!neighborIdx)
					break;
			}
		}

		return res.makeTuple();
	}

	template<typename Dist>
	inline void HnswlibIndex<Dist>::setEf(const size_t ef) {
		this->ef = ef;
	}

	template<typename Dist>
	void bindHnswlibIndex(py::module_& m, const std::string& name) {
		py::class_<HnswlibIndex<Dist>>(m, name.c_str())
			.def(py::init<const SpaceEnum, const size_t>())
			.def("add_items", &HnswlibIndex<Dist>::addItems)
			.def("init_index", &HnswlibIndex<Dist>::init)
			.def("knn_query", &HnswlibIndex<Dist>::knnQuery)
			.def("set_ef", &HnswlibIndex<Dist>::setEf);
	}
}
