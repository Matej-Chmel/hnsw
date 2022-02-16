#include "chm/HnswOptim.hpp"
#include "chm/HnswOrig.hpp"
#include "ChmIndex.hpp"
#include "HnswlibIndex.hpp"

namespace chm {
	PYBIND11_MODULE(hnsw, m) {
		m.doc() = "Python bindings for hnswlib and chm versions of HNSW.";
		bindSpaceEnum(m);
		bindChmIndex<HnswOptim<float>, float>(m, "ChmOptimIndexFloat32");
		bindChmIndex<HnswOrig<float>, float>(m, "ChmOrigIndexFloat32");
		bindHnswlibIndex<float>(m, "HnswlibIndexFloat32");
	}
}
