#include <iostream>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>

namespace py = pybind11;

template<typename T>
void print(const py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
	const auto buf = arr.request();
	const T* const ptr = (const T* const)buf.ptr;

	if(!buf.ndim)
		throw std::runtime_error("No dimensions.");

	const auto dimX = buf.shape[0];

	if(buf.ndim == 1) {
		for(py::ssize_t i = 0; i < dimX; i++)
			std::cout << ptr[i] << ", ";

		std::cout << '\n';
	}
	else if(buf.ndim == 2) {
		const auto dimY = buf.shape[1];

		for(py::ssize_t x = 0; x < dimX; x++) {
			std::cout << '[';

			for(py::ssize_t y = 0; y < dimY; y++)
				std::cout << ptr[x * dimY + y] << ", ";

			std::cout << "]\n";
		}
	} else
		throw std::runtime_error("Too many dimensions.");
}

PYBIND11_MODULE(cpp_module, m) {
	py::add_ostream_redirect(m, "cppStdout");
	m.def("print", print<float>);
}
