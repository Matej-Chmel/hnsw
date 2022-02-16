#pragma once
#include "common.hpp"

namespace chm {
	template<class Algo, typename Dist>
	class ChmIndex {
		std::unique_ptr<Algo> algo;
		const size_t dim;
		DistFunc<Dist> distFunc;
		size_t ef;

	public:
		void addItems(const NumpyArray<Dist> data);
		ChmIndex(const SpaceEnum spaceEnum, const size_t dim);
		void init(const size_t maxElements, const size_t M, const size_t efConstruction, const unsigned int seed);
		py::tuple knnQuery(const NumpyArray<Dist> data, const size_t K);
		void setEf(const size_t ef);
	};

	template<class Algo, typename Dist>
	void bindChmIndex(py::module_& m, const std::string& name);

	template<class Algo, typename Dist>
	inline void ChmIndex<Algo, Dist>::addItems(const NumpyArray<Dist> data) {
		const auto info = getDataInfo(data, this->dim);

		for(size_t i = 0; i < info.count; i++)
			this->algo->insert(info.ptr + i * this->dim);
	}

	template<class Algo, typename Dist>
	inline ChmIndex<Algo, Dist>::ChmIndex(const SpaceEnum spaceEnum, const size_t dim) : algo(nullptr), dim(dim), distFunc{}, ef(DEFAULT_EF) {
		this->distFunc = getDistFunc<Dist>(spaceEnum);
	}

	template<class Algo, typename Dist>
	inline void ChmIndex<Algo, Dist>::init(const size_t maxElements, const size_t M, const size_t efConstruction, const unsigned int seed) {
		this->algo = std::make_unique<Algo>(this->dim, this->distFunc, efConstruction, M, maxElements, seed);
	}

	template<class Algo, typename Dist>
	inline py::tuple ChmIndex<Algo, Dist>::knnQuery(const NumpyArray<Dist> data, const size_t K) {
		return knnQueryImpl(this->algo, data, this->dim, this->ef, K);
	}

	template<class Algo, typename Dist>
	inline void ChmIndex<Algo, Dist>::setEf(const size_t ef) {
		this->ef = ef;
	}

	template<class Algo, typename Dist>
	void bindChmIndex(py::module_& m, const std::string& name) {
		bindIndexImpl<ChmIndex<Algo, Dist>>(m, name);
	}
}
