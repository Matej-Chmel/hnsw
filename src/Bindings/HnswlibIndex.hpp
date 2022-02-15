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
		py::tuple knnQuery(const NumpyArray<Dist> data, const size_t k);
		void setEf(const size_t ef);
	};

	template<typename Dist>
	void bindHnswlibIndex(py::module_& m, const std::string& name);

	template<typename Dist>
	inline void HnswlibIndex<Dist>::addItems(const NumpyArray<Dist> data) {
		const auto buf = data.request();
		checkBufInfo(buf, this->dim);
		const auto count = size_t(buf.shape[0]);
		const Dist* const ptr = (const Dist* const)buf.ptr;

		for(size_t i = 0; i < count; i++)
			this->algo->addPoint(ptr + i * this->dim, i);
	}

	template<typename Dist>
	inline HnswlibIndex<Dist>::HnswlibIndex(const SpaceEnum spaceEnum, const size_t dim) : dim(dim), ef(ef) {
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
	inline py::tuple HnswlibIndex<Dist>::knnQuery(const NumpyArray<Dist> data, const size_t k) {
		const auto buf = data.request();
		checkBufInfo(buf, this->dim);
		const auto count = size_t(buf.shape[0]);
		const auto resLen = count * k;
		const Dist* const ptr = (const Dist* const)buf.ptr;

		auto resDist = new Dist[resLen];
		auto resLabels = new hnswlib::labeltype[resLen];

		for(size_t i = 0; i < count; i++) {
			auto queue = this->algo->searchKnn(ptr + i * this->dim, k);

			for(int resIdx = k - 1; resIdx >= 0; resIdx--) {
				auto& p = queue.top();
				resDist[i * k + resIdx] = p.first;
				resLabels[i * k + resIdx] = p.second;
				queue.pop();
			}
		}

		py::capsule freeDistWhenDone(resDist, freeWhenDone);
		py::capsule freeLabelsWhenDone(resLabels, freeWhenDone);

		return py::make_tuple(
			py::array_t<hnswlib::labeltype>(
				{count, k},
				{k * sizeof(hnswlib::labeltype), sizeof(hnswlib::labeltype)},
				resLabels,
				freeLabelsWhenDone
			),
			py::array_t<Dist>(
				{count, k},
				{k * sizeof(Dist), sizeof(Dist)},
				resDist,
				freeDistWhenDone
			)
		);
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
