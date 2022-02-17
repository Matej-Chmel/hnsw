#include "chm/HnswOptim.hpp"
#include "chm/HnswOrig.hpp"
#include "ChmIndex.hpp"
#include "HnswlibIndex.hpp"
#include "recall.hpp"

namespace chm {
	template<typename Dist>
	void bindDist(py::module_& m, const std::string& typeName) {
		m.def(("getRecall" + typeName).c_str(), getRecall<Dist>);
		bindChmIndex<HnswOptim<Dist>, Dist>(m, ("ChmOptimIndex" + typeName).c_str());
		bindChmIndex<HnswOrig<Dist>, Dist>(m, ("ChmOrigIndex" + typeName).c_str());
		bindHnswlibIndex<hnswlib::BruteforceSearch<Dist>, Dist>(m, ("BruteforceIndex" + typeName).c_str());
		bindHnswlibIndex<hnswlib::HierarchicalNSW<Dist>, Dist>(m, ("HnswlibIndex" + typeName).c_str());
	}

	PYBIND11_MODULE(hnsw, m) {
		m.doc() = "Python bindings for hnswlib and chm versions of HNSW.";
		bindSpaceEnum(m);
		bindDist<float>(m, "Float32");
	}
}
