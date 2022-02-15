#include "ChmOptimIndex.hpp"
#include "ChmOrigIndex.hpp"
#include "HnswlibIndex.hpp"

namespace chm {
	PYBIND11_MODULE(hnsw, m) {
		m.doc() = "Python bindings for hnswlib and chm versions of HNSW.";
		bindSpaceEnum(m);
		bindChmOptimIndex<float>(m, "ChmOptimIndexFloat32");
		bindChmOrigIndex<float>(m, "ChmOrigIndexFloat32");
		bindHnswlibIndex<float>(m, "HnswlibIndexFloat32");
	}
}
