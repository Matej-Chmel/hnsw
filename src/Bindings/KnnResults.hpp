#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace chm {
	namespace py = pybind11;

	template<typename Dist>
	struct DataInfo {
		const size_t count;
		const Dist* const ptr;
	};

	template<typename Dist>
	struct KnnResults {
		const size_t count;
		Dist* dist;
		const size_t K;
		size_t* labels;

		KnnResults(const size_t count, const size_t K);
		py::tuple makeTuple() const;
		void setData(const size_t queryIdx, const size_t neighborIdx, const Dist distance, const size_t label);
	};

	template<typename Dist>
	using NumpyArray = py::array_t<Dist, py::array::c_style | py::array::forcecast>;

	enum class SpaceEnum {
		ANGULAR,
		EUCLIDEAN,
		INNER_PRODUCT
	};

	constexpr auto CONTINUOUS_ERR = "Cannot return the results in a contigious 2D array, ef or M is probably too small.";
	constexpr size_t DEFAULT_EF = 10;
	constexpr auto UNKNOWN_SPACE = "Unknown space";
	constexpr auto WRONG_DIM = "Data must be 1D or 2D array.";
	constexpr auto WRONG_FEATURES = "Number of features doesn't equal to number of dimensions.";

	void bindSpaceEnum(py::module_& m);
	void checkBufInfo(const py::buffer_info& buf, const size_t dim);
	void freeWhenDone(void* d);

	template<typename Dist>
	DataInfo<Dist> getDataInfo(const NumpyArray<Dist>& data, const size_t dim);

	template<typename Dist>
	inline KnnResults<Dist>::KnnResults(const size_t count, const size_t K) : count(count), K(K) {
		const auto len = this->count * this->K;
		this->dist = new Dist[len];
		this->labels = new size_t[len];
	}

	template<typename Dist>
	inline py::tuple KnnResults<Dist>::makeTuple() const {
		return py::make_tuple(
			py::array_t<size_t>(
				{this->count, this->K},
				{this->K * sizeof(size_t), sizeof(size_t)},
				this->labels,
				py::capsule(this->labels, freeWhenDone)
			),
			py::array_t<Dist>(
				{this->count, this->K},
				{this->K * sizeof(Dist), sizeof(Dist)},
				this->dist,
				py::capsule(this->dist, freeWhenDone)
			)
		);
	}

	template<typename Dist>
	inline void KnnResults<Dist>::setData(const size_t queryIdx, const size_t neighborIdx, const Dist distance, const size_t label) {
		this->dist[queryIdx * this->K + neighborIdx] = distance;
		this->labels[queryIdx * this->K + neighborIdx] = label;
	}

	template<typename Dist>
	DataInfo<Dist> getDataInfo(const NumpyArray<Dist>& data, const size_t dim) {
		const auto buf = data.request();
		checkBufInfo(buf, dim);
		return {size_t(buf.shape[0]), (const Dist* const)buf.ptr};
	}
}
