#pragma once
#include "chm/HnswOptim.hpp"
#include "common.hpp"

namespace chm {
	template<typename Dist>
	class ChmOptimIndex {
		std::unique_ptr<HnswOptim<Dist>> algo;
		const size_t dim;
		DistFunc<Dist> distFunc;
		size_t ef;

	public:
		void addItems(const NumpyArray<Dist> data);
		ChmOptimIndex(const SpaceEnum spaceEnum, const size_t dim);
		void init(const size_t maxElements, const size_t M, const size_t efConstruction, const unsigned int seed);
		py::tuple knnQuery(const NumpyArray<Dist> data, const size_t K);
		void setEf(const size_t ef);
	};

	template<typename Dist>
	void bindChmOptimIndex(py::module_& m, const std::string& name);

	template<typename Dist>
	inline void ChmOptimIndex<Dist>::addItems(const NumpyArray<Dist> data) {
		const auto info = getDataInfo(data, this->dim);

		for(size_t i = 0; i < info.count; i++)
			this->algo->insert(info.ptr + i * this->dim);
	}

	template<typename Dist>
	inline ChmOptimIndex<Dist>::ChmOptimIndex(const SpaceEnum spaceEnum, const size_t dim) : algo(nullptr), dim(dim), distFunc{}, ef(DEFAULT_EF) {
		this->distFunc = getDistFunc<Dist>(spaceEnum);
	}

	template<typename Dist>
	inline void ChmOptimIndex<Dist>::init(const size_t maxElements, const size_t M, const size_t efConstruction, const unsigned int seed) {
		this->algo = std::make_unique<HnswOptim<Dist>>(this->dim, this->distFunc, efConstruction, M, maxElements, seed);
	}

	template<typename Dist>
	inline py::tuple ChmOptimIndex<Dist>::knnQuery(const NumpyArray<Dist> data, const size_t K) {
		return knnQueryImpl(this->algo, data, this->dim, this->ef, K);
	}

	template<typename Dist>
	inline void ChmOptimIndex<Dist>::setEf(const size_t ef) {
		this->ef = ef;
	}

	template<typename Dist>
	void bindChmOptimIndex(py::module_& m, const std::string& name) {
		bindIndexImpl<ChmOptimIndex<Dist>>(m, name);
	}
}
