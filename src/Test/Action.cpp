#include <fstream>
#include "Action.hpp"
#include "chm/refImplWrappers.hpp"

namespace chm {
	fs::path getPath(const HNSWAlgoPtr& algo, const std::string& connType, const fs::path& outDir) {
		return outDir / (""_f << algo->getName() << '-' << connType << ".log").str();
	}

	void HNSWAlgoState::buildAndTrack(const FloatVecPtr& coords) {
		{
			std::ofstream s(this->trackConnPath);
			this->conn = this->algo->buildAndTrack(coords, s);
		}

		std::ofstream s(this->finalConnPath);
		writeConnections(this->conn, s);
	}

	const IdxVec3DPtr HNSWAlgoState::getConnections() {
		return this->conn;
	}

	HNSWAlgoState::HNSWAlgoState(HNSWAlgoPtr algo, const fs::path& outDir)
		: algo(algo), conn(nullptr), finalConnPath(getPath(algo, "final", outDir)), trackConnPath(getPath(algo, "track", outDir)) {}

	CommonState::CommonState(const HNSWConfigPtr& cfg, const ElementGenPtr& gen, const fs::path& outDir) : algoStates{
		{
			BacaWrapper::NAME,
			std::make_shared<HNSWAlgoState>(
				std::make_shared<BacaWrapper>(cfg), outDir
			)
		},
		{
			hnswlibWrapper::NAME,
			std::make_shared<HNSWAlgoState>(
				std::make_shared<hnswlibWrapper>(cfg), outDir
			)
		}
	}, cfg(cfg), coords(nullptr), gen(gen), outDir(outDir) {

		ensureDir(this->outDir);
	}

	Action::Action(const std::string& name) : name(name) {}

	ActionResult Action::getActionResult() {
		return {("[DONE] Action \""_f << this->name << "\".").str(), true};
	}

	ActionGenElements::ActionGenElements() : Action("Generate elements") {}

	ActionResult ActionGenElements::run(CommonState* s) {
		s->coords = s->gen->generate();
		return this->getActionResult();
	}

	ActionBuildGraphs::ActionBuildGraphs() : Action("Build graphs") {}

	ActionResult ActionBuildGraphs::run(CommonState* s) {
		for(auto& p : s->algoStates)
			p.second->buildAndTrack(s->coords);
		return this->getActionResult();
	}

	ActionResult Test::getResult(bool passed) {
		return {("["_f << (passed ? "PASSED" : "FAILED") << "] Test \"" << this->name << "\".").str(), passed};
	}

	Test::Test(const std::string& name) : Action(name) {}

	ComparisonTest::ComparisonTest(const std::string& testName, const std::string& algoNameA, const std::string& algoNameB)
		: Test(testName), algoNameA(algoNameA), algoNameB(algoNameB) {}

	ComparedConnections ComparisonTest::getComparedConnections(CommonState* s) {
		return {
			s->algoStates.find(this->algoNameA)->second->getConnections(),
			s->algoStates.find(this->algoNameB)->second->getConnections()
		};
	}

	ActionResult TestLevels::run(CommonState* s) {
		const auto conn = this->getComparedConnections(s);
		const auto len = std::min(conn.A->size(), conn.B->size());

		for(size_t i = 0; i < len; i++)
			if((*conn.A)[i].size() != (*conn.B)[i].size())
				return this->getResult(false);

		return this->getResult(true);
	}

	TestLevels::TestLevels(const std::string& algoNameA, const std::string& algoNameB) : ComparisonTest("Levels", algoNameA, algoNameB) {}

	ActionResult TestNeighborsIndices::run(CommonState* s) {
		const auto conn = this->getComparedConnections(s);
		const auto len = std::min(conn.A->size(), conn.B->size());

		for(size_t i = 0; i < len; i++) {
			const auto& nodeLayersA = (*conn.A)[i];
			const auto& nodeLayersB = (*conn.B)[i];
			const auto nodeLayersLen = std::min(nodeLayersA.size(), nodeLayersB.size());

			for(size_t layerIdx = 0; layerIdx < nodeLayersLen; layerIdx++) {
				const auto& neighborsA = nodeLayersA[layerIdx];
				const auto& neighborsB = nodeLayersB[layerIdx];
				const auto neighborsLen = std::min(neighborsA.size(), neighborsB.size());

				for(size_t neighborIdx = 0; neighborIdx < neighborsLen; neighborIdx++)
					if(neighborsA[neighborIdx] != neighborsB[neighborIdx])
						return this->getResult(false);
			}
		}

		return this->getResult(true);
	}

	TestNeighborsIndices::TestNeighborsIndices(const std::string& algoNameA, const std::string& algoNameB)
		: ComparisonTest("Neighbors indices", algoNameA, algoNameB) {}

	ActionResult TestNeighborsLength::run(CommonState* s) {
		const auto conn = this->getComparedConnections(s);
		const auto len = std::min(conn.A->size(), conn.B->size());

		for(size_t i = 0; i < len; i++) {
			const auto& nodeLayersA = (*conn.A)[i];
			const auto& nodeLayersB = (*conn.B)[i];
			const auto nodeLayersLen = std::min(nodeLayersA.size(), nodeLayersB.size());

			for(size_t layerIdx = 0; layerIdx < nodeLayersLen; layerIdx++)
				if(nodeLayersA[layerIdx].size() != nodeLayersB[layerIdx].size())
					return this->getResult(false);
		}

		return this->getResult(true);
	}

	TestNeighborsLength::TestNeighborsLength(const std::string& algoNameA, const std::string& algoNameB)
		: ComparisonTest("Neighbors length", algoNameA, algoNameB) {}

	ActionResult TestNodeCount::run(CommonState* s) {
		const auto conn = this->getComparedConnections(s);
		return this->getResult(conn.A->size() == s->gen->count && conn.B->size() == s->gen->count);
	}

	TestNodeCount::TestNodeCount(const std::string& algoNameA, const std::string& algoNameB) : ComparisonTest("Node count", algoNameA, algoNameB) {}
}
