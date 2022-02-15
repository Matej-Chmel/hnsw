#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace chm {
	namespace py = pybind11;

	void checkBufInfo(const py::buffer_info& buf, const size_t dim);

	constexpr auto CONTINUOUS_ERR = "Cannot return the results in a contigious 2D array, ef or M is probably too small.";

	void freeWhenDone(void* d);

	template<typename Dist>
	using NumpyArray = py::array_t<Dist, py::array::c_style | py::array::forcecast>;

	constexpr auto UNKNOWN_SPACE = "Unknown space";
	constexpr auto WRONG_DIM = "Data must be 1D or 2D array.";
	constexpr auto WRONG_FEATURES = "Number of features doesn't equal to number of dimensions.";
}
