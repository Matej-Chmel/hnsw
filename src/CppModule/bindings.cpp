#include <pybind11/pybind11.h>

namespace py = pybind11;

int add(const int a, const int b) {
	return a + b;
}

PYBIND11_MODULE(cpp_module, m) {
	m.def("add", add);
}
