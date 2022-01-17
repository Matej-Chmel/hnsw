#include <fstream>
#include "Action.hpp"
#include "chm/DebugHNSWRunner.hpp"
#include "chm/debugWrappers.hpp"

namespace chm {
	void throwUnknownKind(HNSWAlgoKind k) {
		throw AppError("Unknown algo kind: "_f << int(k) << '.');
	}

	HNSWAlgoStatePtr getHNSWAlgoState(HNSWAlgoKind k, const HNSWConfigPtr& cfg, const fs::path& outDir) {
		HNSWAlgoPtr algo;

		switch(k) {
			case HNSWAlgoKind::BACA:
				algo = std::make_shared<BacaWrapper>(cfg);
				break;
			case HNSWAlgoKind::BACA_DEBUG:
				algo = std::make_shared<BacaDebugWrapper>(cfg);
				break;
			case HNSWAlgoKind::HNSWLIB:
				algo = std::make_shared<hnswlibWrapper>(cfg);
				break;
			case HNSWAlgoKind::HNSWLIB_DEBUG:
				algo = std::make_shared<hnswlibDebugWrapper>(cfg);
				break;
			default:
				throwUnknownKind(k);
		}

		return std::make_shared<HNSWAlgoState>(algo, outDir);
	}

	fs::path getPath(const HNSWAlgoPtr& algo, const std::string& connType, const fs::path& outDir) {
		return outDir / (""_f << algo->getName() << '-' << connType << ".log").str();
	}

	void HNSWAlgoState::writeFinalConn() {
		std::ofstream s(this->finalConnPath);
		writeConnections(this->conn, s);
	}

	void HNSWAlgoState::build(const FloatVecPtr& coords) {
		this->algo->build(coords);
		this->saveConnections();
	}

	void HNSWAlgoState::buildAndTrack(const FloatVecPtr& coords) {
		{
			std::ofstream s(this->trackConnPath);
			this->conn = this->algo->buildAndTrack(coords, s);
		}
		this->writeFinalConn();
	}

	const IdxVec3DPtr HNSWAlgoState::getConnections() const {
		return this->conn;
	}

	std::string HNSWAlgoState::getName() const {
		return this->algo->getName();
	}

	HNSWAlgoState::HNSWAlgoState(HNSWAlgoPtr algo, const fs::path& outDir)
		: conn(nullptr), finalConnPath(getPath(algo, "final", outDir)), trackConnPath(getPath(algo, "track", outDir)), algo(algo) {}

	void HNSWAlgoState::saveConnections() {
		this->conn = this->algo->getConnections();
		this->writeFinalConn();
	}

	CommonState::CommonState(
		const HNSWConfigPtr& cfg, const ElementGenPtr& gen, const std::vector<HNSWAlgoKind>& algoKinds, const fs::path& outDir
	) : cfg(cfg), coords(nullptr), gen(gen), outDir(outDir) {

		for(auto& k : algoKinds)
			this->algoStates.push_back(getHNSWAlgoState(k, this->cfg, this->outDir));
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

	ActionBuildGraphs::ActionBuildGraphs(bool track) : Action("Build graphs"), track(track) {}

	ActionResult ActionBuildGraphs::run(CommonState* s) {
		for(auto& state : s->algoStates)
			if(this->track)
				state->buildAndTrack(s->coords);
			else
				state->build(s->coords);

		return this->getActionResult();
	}

	ActionResult Test::getResult(bool passed) {
		return {("["_f << (passed ? "PASSED" : "FAILED") << "] Test \"" << this->name << "\".").str(), passed};
	}

	Test::Test(const std::string& name) : Action(name) {}

	const HNSWAlgoStatePtr ComparisonTest::getA(const CommonState* const s) const {
		return s->algoStates[this->idxA];
	}

	const HNSWAlgoStatePtr ComparisonTest::getB(const CommonState* const s) const {
		return s->algoStates[this->idxB];
	}

	ComparisonTest::ComparisonTest(const std::string& testName, size_t idxA, size_t idxB) : Test(testName), idxA(idxA), idxB(idxB) {}

	ComparedConnections ComparisonTest::getComparedConnections(CommonState* s) {
		return {this->getA(s)->getConnections(), this->getB(s)->getConnections()};
	}

	std::string ComparisonTest::getNames(CommonState* s) {
		return ("\t"_f <<
			this->getA(s)->getName() << "\n\t" <<
			this->getB(s)->getName() << '\n'
		).str();
	}

	ActionResult TestLevels::run(CommonState* s) {
		const auto conn = this->getComparedConnections(s);
		const auto len = std::min(conn.A->size(), conn.B->size());

		for(size_t i = 0; i < len; i++)
			if((*conn.A)[i].size() != (*conn.B)[i].size())
				return this->getResult(false);

		return this->getResult(true);
	}

	TestLevels::TestLevels(size_t idxA, size_t idxB) : ComparisonTest("Levels", idxA, idxB) {}

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

	TestNeighborsIndices::TestNeighborsIndices(size_t idxA, size_t idxB) : ComparisonTest("Neighbors indices", idxA, idxB) {}

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

	TestNeighborsLength::TestNeighborsLength(size_t idxA, size_t idxB) : ComparisonTest("Neighbors length", idxA, idxB) {}

	ActionResult TestNodeCount::run(CommonState* s) {
		const auto conn = this->getComparedConnections(s);
		return this->getResult(conn.A->size() == s->gen->count && conn.B->size() == s->gen->count);
	}

	TestNodeCount::TestNodeCount(size_t idxA, size_t idxB) : ComparisonTest("Node count", idxA, idxB) {}

	ActionResult TestConnections::run(CommonState* s) {
		ActionResult res{("\nComparing connections between:\n"_f << this->getNames(s)).str(), true};
		std::vector<ActionPtr> tests{
			std::make_shared<TestNodeCount>(this->idxA, this->idxB),
			std::make_shared<TestLevels>(this->idxA, this->idxB),
			std::make_shared<TestNeighborsLength>(this->idxA, this->idxB),
			std::make_shared<TestNeighborsIndices>(this->idxA, this->idxB)
		};

		for(auto& t : tests) {
			auto testRes = t->run(s);
			res.msg += '\n' + testRes.msg;

			if(!testRes.success)
				res.success = false;
		}

		res.msg += '\n';
		return res;
	}

	TestConnections::TestConnections(size_t idxA, size_t idxB) : ComparisonTest("Connections", idxA, idxB) {}

	ActionDebugBuild::ActionDebugBuild() : Action("Build graphs with debug interface") {}

	ActionResult ActionDebugBuild::run(CommonState* s) {
		HNSWAlgoVec debugAlgos;

		for(auto& state : s->algoStates)
			debugAlgos.push_back(state->algo);

		std::ofstream errStream(s->outDir / "debug-build.log");
		DebugHNSWRunner runner(debugAlgos, errStream);

		runner.build(s->coords, s->cfg->dim);

		for(auto& state : s->algoStates)
			state->saveConnections();

		return this->getActionResult();
	}
}
