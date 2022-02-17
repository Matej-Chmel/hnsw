#pragma once
#include <unordered_set>
#include "KnnResults.hpp"

namespace chm {
	template<typename T>
	struct NumpyArrayWrapper {
		const py::buffer_info buf;
		const T* const data;
		const size_t xDim;
		const size_t yDim;

		void fillSet(std::unordered_set<T>& set, const size_t x) const;
		const T& get(const size_t x, const size_t y) const;
		size_t getComponentCount() const;
		NumpyArrayWrapper(const NumpyArray<T>& a);
	};

	template<typename Dist>
	float getRecall(const NumpyArray<Dist> correctLabels, const NumpyArray<Dist> labels);

	template<typename T>
	inline void NumpyArrayWrapper<T>::fillSet(std::unordered_set<T>& set, const size_t x) const {
		set.clear();

		for(size_t y = 0; y < this->yDim; y++)
			set.insert(this->get(x, y));
	}

	template<typename T>
	inline const T& NumpyArrayWrapper<T>::get(const size_t x, const size_t y) const {
		return this->data[x * this->yDim + y];
	}

	template<typename T>
	inline size_t NumpyArrayWrapper<T>::getComponentCount() const {
		return this->xDim * this->yDim;
	}

	template<typename T>
	inline NumpyArrayWrapper<T>::NumpyArrayWrapper(const NumpyArray<T>& a)
		: buf(a.request()), data((const T* const)this->buf.ptr), xDim(this->buf.shape[0]), yDim(this->buf.shape[1]) {}

	template<typename Dist>
	float getRecall(const NumpyArray<Dist> correctLabels, const NumpyArray<Dist> testedLabels) {
		NumpyArrayWrapper correctWrapper(correctLabels), testedWrapper(testedLabels);
		size_t hits = 0;
		std::unordered_set<Dist> correctSet;
		correctSet.reserve(correctWrapper.yDim);

		for(size_t x = 0; x < correctWrapper.xDim; x++) {
			correctWrapper.fillSet(correctSet, x);

			for(size_t y = 0; y < correctWrapper.yDim; y++)
				if(correctSet.find(testedWrapper.get(x, y)) != correctSet.end())
					hits++;
		}

		return float(hits) / (correctWrapper.getComponentCount());
	}
}
