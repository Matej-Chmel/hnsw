#include "pybind.hpp"

PYBIND11_MODULE(hnsw, m) {
	m.doc() = "Python bindings for hnswlib and chm versions of HNSW.";
}
