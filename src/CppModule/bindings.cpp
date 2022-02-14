#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

template<typename T>
void multiply(const py::array_t<T, py::array::c_style | py::array::forcecast> arr, const T n) {
	const auto buf = arr.request();
	T* const ptr = (T* const)buf.ptr;

	if(!buf.ndim)
		throw std::runtime_error("No dimensions.");

	const auto dimX = buf.shape[0];

	if(buf.ndim == 1)
		for(py::ssize_t i = 0; i < dimX; i++)
			ptr[i] *= n;
	else if(buf.ndim == 2) {
		const auto dimY = buf.shape[1];

		for(py::ssize_t x = 0; x < dimX; x++)
			for(py::ssize_t y = 0; y < dimY; y++)
				ptr[x * dimY + y] *= n;
	} else
		throw std::runtime_error("Too many dimensions.");
}

PYBIND11_MODULE(cpp_module, m) {
	m.def("multiply", multiply<float>);
}
