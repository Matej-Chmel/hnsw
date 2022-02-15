#include "ChmOrigIndex.hpp"
#include "HnswlibIndex.hpp"

namespace chm {
	PYBIND11_MODULE(hnsw, m) {
		m.doc() = "Python bindings for hnswlib and chm versions of HNSW.";
		bindSpaceEnum(m);
		bindChmOrigIndex<float>(m, "ChmOrigIndexFloat32");
		bindHnswlibIndex<float>(m, "HnswlibIndexFloat32");
	}
}
