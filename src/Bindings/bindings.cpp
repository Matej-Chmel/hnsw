#include "HnswlibIndex.hpp"

namespace chm {
	PYBIND11_MODULE(hnsw, m) {
		m.doc() = "Python bindings for hnswlib and chm versions of HNSW.";
		bindSpaceEnum(m);
		bindHnswlibIndex<float>(m, "HnswlibIndexFloat");
	}
}
