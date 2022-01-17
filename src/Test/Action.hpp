#pragma once
#include "chm/KNNAlgo.hpp"

namespace chm {
	class HNSWAlgoState : public Unique {
		IdxVec3DPtr conn;
		fs::path finalConnPath;
		fs::path trackConnPath;

		void writeFinalConn();

	public:
		HNSWAlgoPtr algo;

		void build(const FloatVecPtr& coords);
		void buildAndTrack(const FloatVecPtr& coords);
		const IdxVec3DPtr getConnections() const;
		std::string getName() const;
		HNSWAlgoState(HNSWAlgoPtr algo, const fs::path& outDir);
		void saveConnections();
	};

	typedef std::shared_ptr<HNSWAlgoState> HNSWAlgoStatePtr;
	typedef std::vector<HNSWAlgoStatePtr> HNSWAlgoStateVec;

	enum class HNSWAlgoKind {
		BACA,
		BACA_DEBUG,
		HNSWLIB,
		HNSWLIB_DEBUG
	};

	struct CommonState : public Unique {
		HNSWAlgoStateVec algoStates;
		const HNSWConfigPtr cfg;
		FloatVecPtr coords;
		const ElementGenPtr gen;
		const fs::path outDir;

		CommonState(const HNSWConfigPtr& cfg, const ElementGenPtr& gen, const std::vector<HNSWAlgoKind>& algoKinds, const fs::path& outDir);
	};

	struct ActionResult {
		std::string msg;
		bool success;
	};

	class Action {
	protected:
		std::string name;

		Action(const std::string& name);
		ActionResult getActionResult();

	public:
		virtual ~Action() = default;
		virtual ActionResult run(CommonState* s) = 0;
	};

	typedef std::shared_ptr<Action> ActionPtr;

	class ActionGenElements : public Action {
	public:
		ActionGenElements();
		ActionResult run(CommonState* s) override;
	};

	class ActionBuildGraphs : public Action {
		const bool track;

	public:
		ActionBuildGraphs(bool track);
		ActionResult run(CommonState* s) override;
	};

	class Test : public Action {
	protected:
		ActionResult getResult(bool passed);
		Test(const std::string& name);

	public:
		virtual ~Test() = default;
		virtual ActionResult run(CommonState* s) = 0;
	};

	struct ComparedConnections {
		const IdxVec3DPtr A, B;
	};

	class ComparisonTest : public Test {
		const HNSWAlgoStatePtr getA(const CommonState* const s) const;
		const HNSWAlgoStatePtr getB(const CommonState* const s) const;

	protected:
		size_t idxA, idxB;

		ComparisonTest(const std::string& testName, size_t idxA, size_t idxB);
		ComparedConnections getComparedConnections(CommonState* s);

	public:
		std::string getNames(CommonState* s);
	};

	class TestLevels : public ComparisonTest {
	public:
		ActionResult run(CommonState* s) override;
		TestLevels(size_t idxA, size_t idxB);
	};

	class TestNeighborsIndices : public ComparisonTest {
	public:
		ActionResult run(CommonState* s) override;
		TestNeighborsIndices(size_t idxA, size_t idxB);
	};

	class TestNeighborsLength : public ComparisonTest {
	public:
		ActionResult run(CommonState* s) override;
		TestNeighborsLength(size_t idxA, size_t idxB);
	};

	class TestNodeCount : public ComparisonTest {
	public:
		ActionResult run(CommonState* s) override;
		TestNodeCount(size_t idxA, size_t idxB);
	};

	class TestConnections : public ComparisonTest {
	public:
		ActionResult run(CommonState* s) override;
		TestConnections(size_t idxA, size_t idxB);
	};

	class ActionDebugBuild : public Action {
	public:
		ActionDebugBuild();
		ActionResult run(CommonState* s) override;
	};
}
