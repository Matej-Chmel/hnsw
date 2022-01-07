#pragma once
#include <string>
#include <unordered_map>
#include "chm/KNNAlgo.hpp"

namespace chm {
	class HNSWAlgoState : public Unique {
		HNSWAlgoPtr algo;
		IdxVec3DPtr conn;
		fs::path finalConnPath;
		fs::path trackConnPath;

	public:
		void buildAndTrack(const FloatVecPtr& coords);
		const IdxVec3DPtr getConnections();
		HNSWAlgoState(HNSWAlgoPtr algo, const fs::path& outDir);
	};

	typedef std::shared_ptr<HNSWAlgoState> HNSWAlgoStatePtr;

	struct CommonState : public Unique {
		const std::unordered_map<std::string, HNSWAlgoStatePtr> algoStates;
		const HNSWConfigPtr cfg;
		FloatVecPtr coords;
		const ElementGenPtr gen;
		const fs::path outDir;

		CommonState(const HNSWConfigPtr& cfg, const ElementGenPtr& gen, const fs::path& outDir);
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
	public:
		ActionBuildGraphs();
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
		std::string algoNameA, algoNameB;

	protected:
		ComparisonTest(const std::string& testName, const std::string& algoNameA, const std::string& algoNameB);
		ComparedConnections getComparedConnections(CommonState* s);
	};

	class TestLevels : public ComparisonTest {
	public:
		ActionResult run(CommonState* s) override;
		TestLevels(const std::string& algoNameA, const std::string& algoNameB);
	};

	class TestNeighborsIndices : public ComparisonTest {
	public:
		ActionResult run(CommonState* s) override;
		TestNeighborsIndices(const std::string& algoNameA, const std::string& algoNameB);
	};

	class TestNeighborsLength : public ComparisonTest {
	public:
		ActionResult run(CommonState* s) override;
		TestNeighborsLength(const std::string& algoNameA, const std::string& algoNameB);
	};

	class TestNodeCount : public ComparisonTest {
	public:
		ActionResult run(CommonState* s) override;
		TestNodeCount(const std::string& algoNameA, const std::string& algoNameB);
	};
}
