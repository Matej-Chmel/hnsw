#pragma once
#include <hnswlib/templateSpaces.hpp>
#include "common.hpp"

namespace chm {
	template<typename Algo, typename Dist>
	class HnswlibIndex {
		std::unique_ptr<Algo> algo;
		const size_t dim;
		bool normalize;
		std::vector<Dist> normCoords;
		std::unique_ptr<hnswlib::SpaceInterface<Dist>> space;

	public:
		void addItems(const NumpyArray<Dist> data);
		HnswlibIndex(const SpaceEnum spaceEnum, const size_t dim);
		void init(const size_t maxElements, const size_t M, const size_t efConstruction, const unsigned int seed);
		py::tuple knnQuery(const NumpyArray<Dist> data, const size_t K);
		void setEf(const size_t ef);
	};

	template<typename Algo, typename Dist>
	void bindHnswlibIndex(py::module_& m, const std::string& name);

	template<typename Algo, typename Dist>
	inline void HnswlibIndex<Algo, Dist>::addItems(const NumpyArray<Dist> data) {
		const auto info = getDataInfo(data, this->dim);

		if(this->normalize)
			for(size_t i = 0; i < info.count; i++) {
				normalizeData(info.ptr + i * this->dim, this->normCoords);
				this->algo->addPoint(this->normCoords.data(), i);
			}
		else
			for(size_t i = 0; i < info.count; i++)
				this->algo->addPoint(info.ptr + i * this->dim, i);
	}

	template<typename Algo, typename Dist>
	inline HnswlibIndex<Algo, Dist>::HnswlibIndex(const SpaceEnum spaceEnum, const size_t dim) : algo(nullptr), dim(dim), normalize(false) {
		switch(spaceEnum) {
			case SpaceEnum::ANGULAR:
				this->normalize = true;
				this->normCoords.resize(this->dim);
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

	template<typename Algo, typename Dist>
	inline void HnswlibIndex<Algo, Dist>::init(const size_t maxElements, const size_t M, const size_t efConstruction, const unsigned int seed) {
		if constexpr(std::is_same<Algo, hnswlib::BruteforceSearch<Dist>>::value)
			this->algo = std::make_unique<Algo>(this->space.get(), maxElements);
		else
			this->algo = std::make_unique<Algo>(this->space.get(), maxElements, M, efConstruction, seed);
	}

	template<typename Algo, typename Dist>
	inline py::tuple HnswlibIndex<Algo, Dist>::knnQuery(const NumpyArray<Dist> data, const size_t K) {
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

	template<typename Algo, typename Dist>
	inline void HnswlibIndex<Algo, Dist>::setEf(const size_t ef) {
		if constexpr(std::is_same<Algo, hnswlib::HierarchicalNSW<Dist>>::value)
			this->algo->setEf(ef);
	}

	template<typename Algo, typename Dist>
	void bindHnswlibIndex(py::module_& m, const std::string& name) {
		bindIndexImpl<HnswlibIndex<Algo, Dist>>(m, name);
	}
}
