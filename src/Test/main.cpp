#include <cstdlib>
#include <fstream>
#include <iostream>
#include "chm/AppError.hpp"
#include "chm/literals.hpp"
#include "chm/refImplWrappers.hpp"
namespace fs = chm::fs;
using namespace chm::literals;

constexpr size_t EF_CONSTRUCTION = 200;
constexpr size_t DIM = 128;
constexpr auto ELEMENT_MAX = 1.f;
constexpr auto ELEMENT_MIN = 0.f;
constexpr unsigned int ELEMENT_SEED = 100;
constexpr unsigned int LEVEL_SEED = 100;
constexpr size_t MAX_NEIGHBORS = 16;
constexpr size_t NODE_COUNT = 100;

chm::FloatVecPtr getCoords() {
	return chm::ElementGenerator(NODE_COUNT, DIM, ELEMENT_MIN, ELEMENT_MAX, ELEMENT_SEED).generate();
}

fs::path getLogsDir() {
	const auto logsDir = fs::path(SOLUTION_DIR) / "logs";
	chm::ensureDir(logsDir);
	return logsDir;
}

void testFailed(const chm::HNSWAlgorithm& a, const chm::HNSWAlgorithm& b, const std::stringstream& reason) {
	throw chm::AppError("A: "_f << a.getInfo() << "\nB: " << b.getInfo() << "\nReason:\n" << reason.str());
}

void testConnections(const chm::HNSWAlgorithm& a, const chm::HNSWAlgorithm& b, const chm::IdxVec3D& aConn, const chm::IdxVec3D& bConn) {
	if(aConn.size() != bConn.size())
		testFailed(a, b, "Node count mismatch.\nA: "_f << aConn.size() << "\nB: " << bConn.size());

	const auto nodeCount = aConn.size();

	for(size_t nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++) {
		const auto& aNodeLayers = aConn[nodeIdx];
		const auto& bNodeLayers = bConn[nodeIdx];

		if(aNodeLayers.size() != bNodeLayers.size())
			testFailed(a, b, "Level mismatch for node "_f << nodeIdx << "\nA: " << aNodeLayers.size() - 1 << "\nB: " << bNodeLayers.size() - 1);

		const auto nodeLayersLen = aNodeLayers.size();

		for(size_t layerIdx = 0; layerIdx < nodeLayersLen; layerIdx++) {
			const auto& aNeighbors = aNodeLayers[layerIdx];
			const auto& bNeighbors = bNodeLayers[layerIdx];

			if(aNeighbors.size() != bNeighbors.size())
				testFailed(
					a, b, "Neighbor count mismatch for node "_f << nodeIdx << " at level " << layerIdx << "\nA: " << aNeighbors.size() <<
					"\nB: " << bNeighbors.size()
				);

			const auto neighborsLen = aNeighbors.size();

			for(size_t neighborIdx = 0; neighborIdx < neighborsLen; neighborIdx++)
				if(aNeighbors[neighborIdx] != bNeighbors[neighborIdx])
					testFailed(a, b, "Neighbor ID mismatch for node"_f << nodeIdx << " at level " << layerIdx << " at index " << neighborIdx <<
						"\nA: " << aNeighbors[neighborIdx] << "\nB: " << bNeighbors[neighborIdx]
					);
		}
	}
}

void writeConnections(const chm::HNSWAlgorithm& algo, const chm::IdxVec3DPtr& conn, const fs::path& logsDir) {
	const auto path = logsDir / (algo.getInfo() + ".log");
	std::ofstream s(path);
	chm::writeConnections(conn, s);
}

int main() {
	try {
		const auto cfg = std::make_shared<chm::HNSWConfig>(DIM, EF_CONSTRUCTION, MAX_NEIGHBORS, NODE_COUNT, LEVEL_SEED);
		chm::BacaWrapper bacaAlgo(cfg);
		chm::hnswlibWrapper hnswlibAlgo(cfg);

		const auto coords = getCoords();
		bacaAlgo.build(coords);
		hnswlibAlgo.build(coords);

		const auto bacaConn = bacaAlgo.getConnections();
		const auto hnswlibConn = hnswlibAlgo.getConnections();

		try {
			testConnections(bacaAlgo, hnswlibAlgo, *bacaConn, *hnswlibConn);
			std::cout << "Test PASSED.\n";

		} catch(chm::AppError& e) {
			std::cerr << "[Test FAILED]\n" << e.what() << '\n';
			std::cout << "Writing connections.\n";

			const auto logsDir = getLogsDir();
			writeConnections(bacaAlgo, bacaConn, logsDir);
			writeConnections(hnswlibAlgo, hnswlibConn, logsDir);
			std::cout << "Connections written.\n";

			return EXIT_FAILURE;
		}

	} catch(chm::AppError& e) {
		std::cerr << "[ERROR] " << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
