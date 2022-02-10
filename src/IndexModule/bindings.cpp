#define CHM_HNSW_INTERMEDIATE
#define DECIDE_BY_IDX
#include <pybind11/chrono.h>
#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "IndexLib/Index.hpp"
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<float>);

template<typename Coord>
void defineModule(py::module_& m) {
	using namespace chm;

	py::class_<QueryTime>(m, "QueryTime")
		.def_readonly("accumulated", &QueryTime::accumulated)
		.def_readonly("avg", &QueryTime::avg)
		.def_readonly("queries", &QueryTime::queries);

	py::class_<AlgoBuildRes>(m, "AlgoBuildRes")
		.def_readonly("init", &AlgoBuildRes::init)
		.def_readonly("queryTime", &AlgoBuildRes::queryTime);

	py::class_<SeqAlgoBuildRes, IndexBuildResPtr>(m, "IndexBuildRes")
		.def_readonly("init", &AlgoBuildRes::init)
		.def_readonly("queryTime", &AlgoBuildRes::queryTime)
		.def_readonly("total", &SeqAlgoBuildRes::total);

	py::class_<FoundNeighbors<Coord>, FoundNeighborsPtr<Coord>>(m, "FoundNeighbors")
		.def_readonly("distances", &FoundNeighbors<Coord>::distances)
		.def_readonly("indices", &FoundNeighbors<Coord>::indices);

	py::class_<AlgoSearchRes<Coord>>(m, "AlgoSearchRes")
		.def_readonly("queryTime", &AlgoSearchRes<Coord>::queryTime)
		.def_readonly("recall", &AlgoSearchRes<Coord>::recall);

	py::class_<SeqAlgoSearchRes<Coord>, IndexSearchResPtr<Coord>>(m, "IndexSearchRes")
		.def_readonly("queryTime", &SeqAlgoSearchRes<Coord>::queryTime)
		.def_readonly("recall", &SeqAlgoSearchRes<Coord>::recall)
		.def_readonly("total", &SeqAlgoSearchRes<Coord>::total);

	py::class_<HnswCfg, HnswCfgPtr>(m, "HnswCfg")
		.def(py::init<const size_t, const size_t, const size_t, const size_t, const unsigned int, const bool>());

	py::enum_<HnswKind>(m, "HnswKind")
		.value("CHM_AUTO", HnswKind::CHM_AUTO)
		.value("HNSWLIB", HnswKind::HNSWLIB);

	py::class_<HnswSettings, HnswSettingsPtr>(m, "HnswSettings")
		.def(py::init<const bool, const bool, const bool, const bool>());

	py::class_<HnswType, HnswTypePtr>(m, "HnswType")
		.def(py::init<const HnswCfgPtr&, const HnswKind, const HnswSettingsPtr&>());

	py::class_<ICoords<Coord>, ICoordsPtr<Coord>>(m, "ICoords")
		.def("getCount", &ICoords<Coord>::getCount);

	py::class_<SearchCfg<Coord>, SearchCfgPtr<Coord>>(m, "SearchCfg")
		.def(py::init<const ICoordsPtr<Coord>&, const size_t, const size_t>());

	py::class_<Index<Coord>, IndexPtr<Coord>>(m, "Index")
		.def(py::init<const HnswTypePtr&>())
		.def("build", &Index<Coord>::build)
		.def("search", &Index<Coord>::search);

	m.def("bruteforce", [](const VecPtr<Coord>& nodes, const VecPtr<Coord>& queries, const size_t dim, const size_t K, const bool useEuclid) {
		if(useEuclid)
			return bruteforce<Coord, true>(nodes, queries, dim, K);
		return bruteforce<Coord, false>(nodes, queries, dim, K);
	});
	m.def("RndCoords", getRndCoords<Coord>);
	m.doc() = "Custom implementation of HNSW graph.";

	py::add_ostream_redirect(m, "cppStdout");
}

PYBIND11_MODULE(IndexModule, m) {
	defineModule<float>(m);
}
