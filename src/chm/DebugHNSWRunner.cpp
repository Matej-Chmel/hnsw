#include <fstream>
#include "DebugHNSWRunner.hpp"

namespace chm {
	auto TEST_LEVEL = "Last generated level";
	auto TEST_NEAREST_NODE = "Nearest node after upper search";
	auto TEST_LOWER_SEARCH = "Search of lower layers";
	auto TEST_SELECT_NEIGHBORS = "Selecting neighbors";
	auto TEST_CONNECTIONS = "Connections";
	auto TEST_ENTRY = "Entry point";

	void writeIdxVec(const IdxVecPtr& vec, std::ostream& stream) {
		auto& v = *vec;
		const auto len = v.size();

		stream << "[Length " << len << "]\n\n";

		for(size_t i = 0; i < len; i++)
			stream << "[" << i << "]: " << v[i] << '\n';
	}

	void writeNodeVec(const NodeVecPtr& vec, std::ostream& stream) {
		auto& v = *vec;
		const auto len = v.size();

		stream << "[Length " << len << "]\n\n";

		for(size_t i = 0; i < len; i++) {
			auto& node = v[i];
			stream << "[" << i << "]: " << node.idx << ", " << node.distance << '\n';
		}
	}

	fs::path DebugHNSWRunner::actualLogPath() const {
		return this->outDir / "actual.log";
	}

	std::string DebugHNSWRunner::algoName(size_t i) const {
		return this->algos[i]->getName();
	}

	DebugHNSW* DebugHNSWRunner::debugObj(size_t i) {
		return this->algos[i]->getDebugObject();
	}

	fs::path DebugHNSWRunner::expectedLogPath() const {
		return this->outDir / "expected.log";
	}

	void DebugHNSWRunner::insert(float* coords, size_t idx) {
		this->currentIdx = idx;

		this->startInsert(coords, idx);
		this->testLatestLevel();
		this->prepareUpperSearch();

		auto range = this->getUpperRange();

		if(range.shouldLoop)
			for(auto lc = range.start; lc > range.end; lc--) {
				this->searchUpperLayers(lc);
				this->testNearestNode(lc);
			}

		range = this->getLowerRange();

		if(range.shouldLoop) {
			this->prepareLowerSearch();

			for(auto lc = range.start;; lc--) {
				this->searchLowerLayers(lc);
				this->testLowerLayerResults(lc);

				this->selectOriginalNeighbors(lc);
				auto neighbors = this->testOriginalNeighbors(lc);

				this->connect(lc);
				this->testConnections(neighbors, idx, lc);
				this->prepareNextLayer(lc);

				if(lc == 0)
					break;
			}
		}

		this->setupEnterPoint();
		this->testEnterPoint();
	}

	size_t DebugHNSWRunner::len() const {
		return this->algos.size();
	}

	fs::path DebugHNSWRunner::reportLogPath() const {
		return this->outDir / "report.log";
	}

	void DebugHNSWRunner::throwAppError(const char* testName, size_t algoIdx) const {
		throw AppError(
			"Algorithm "_f << this->algoName(algoIdx) << " failed test \"" << testName <<
			"\" when inserting node " << this->currentIdx << '.'
		);
	}

	void DebugHNSWRunner::writeReport(const char* testName, size_t algoIdx, std::ostream& stream, bool useLayer, size_t lc) {
		stream <<
			"Error report\nReference algorithm: " << this->algoName() << "\nWrong algorithm: " << this->algoName(algoIdx) <<
			"\nFailed test: " << testName << "\nNode: " << this->currentIdx;

		if(useLayer)
			stream << "\nLayer: " << lc;

		stream << '\n';
	}

	void DebugHNSWRunner::startInsert(float* coords, size_t idx) {
		for(auto& a : this->algos)
			a->getDebugObject()->startInsert(coords, idx);
	}

	void DebugHNSWRunner::testLatestLevel() {
		auto expectedLevel = this->debugObj()->getLatestLevel();
		auto len = this->len();

		for(size_t i = 1; i < len; i++) {
			auto actualLevel = this->debugObj(i)->getLatestLevel();

			if(actualLevel != expectedLevel) {
				{
					std::ofstream stream(this->reportLogPath());
					this->writeReport(TEST_LEVEL, i, stream);
				}
				{
					std::ofstream stream(this->expectedLogPath());
					stream << "Level: " << expectedLevel << '\n';
				}
				{
					std::ofstream stream(this->actualLogPath());
					stream << "Level: " << actualLevel << '\n';
				}
				this->throwAppError(TEST_LEVEL, i);
			}
		}
	}

	void DebugHNSWRunner::prepareUpperSearch() {
		for(auto& a : this->algos)
			a->getDebugObject()->prepareUpperSearch();
	}

	LevelRange DebugHNSWRunner::getUpperRange() {
		return this->algos[0]->getDebugObject()->getUpperRange();
	}

	void DebugHNSWRunner::searchUpperLayers(size_t lc) {
		for(auto& a : this->algos)
			a->getDebugObject()->searchUpperLayers(lc);
	}

	void DebugHNSWRunner::testNearestNode(size_t lc) {
		auto expectedNode = this->debugObj()->getNearestNode();
		auto len = this->len();

		for(size_t i = 1; i < len; i++) {
			auto actualNode = this->debugObj(i)->getNearestNode();

			if(actualNode.distance != expectedNode.distance || actualNode.idx != expectedNode.idx) {
				{
					std::ofstream stream(this->reportLogPath());
					this->writeReport(TEST_NEAREST_NODE, i, stream, true, lc);
				}
				{
					std::ofstream stream(this->expectedLogPath());
					stream << "Distance: " << expectedNode.distance << "\nIndex: " << expectedNode.idx << '\n';
				}
				{
					std::ofstream stream(this->actualLogPath());
					stream << "Distance: " << actualNode.distance << "\nIndex: " << actualNode.idx << '\n';
				}
				this->throwAppError(TEST_NEAREST_NODE, i);
			}
		}
	}

	void DebugHNSWRunner::prepareLowerSearch() {
		for(auto& a : this->algos)
			a->getDebugObject()->prepareLowerSearch();
	}

	LevelRange DebugHNSWRunner::getLowerRange() {
		return this->algos[0]->getDebugObject()->getLowerRange();
	}

	void DebugHNSWRunner::searchLowerLayers(size_t lc) {
		for(auto& a : this->algos)
			a->getDebugObject()->searchLowerLayers(lc);
	}

	void DebugHNSWRunner::testLowerLayerResults(size_t lc) {
		auto expectedRes = this->debugObj()->getLowerLayerResults();
		auto& expectedVec = *expectedRes;
		auto expectedLen = expectedVec.size();
		auto algosLen = this->len();

		auto fail = [&](const NodeVecPtr& actualRes, size_t algoIdx) {
			{
				std::ofstream stream(this->reportLogPath());
				this->writeReport(TEST_LOWER_SEARCH, algoIdx, stream, true, lc);
			}
			{
				std::ofstream stream(this->expectedLogPath());
				writeNodeVec(expectedRes, stream);
			}
			{
				std::ofstream stream(this->actualLogPath());
				writeNodeVec(actualRes, stream);
			}
			this->throwAppError(TEST_LOWER_SEARCH, algoIdx);
		};

		for(size_t algoIdx = 1; algoIdx < algosLen; algoIdx++) {
			auto actualRes = this->debugObj(algoIdx)->getLowerLayerResults();
			auto& actualVec = *actualRes;
			auto actualLen = actualVec.size();

			if(actualLen != expectedLen)
				fail(actualRes, algoIdx);

			for(size_t i = 0; i < expectedLen; i++) {
				auto& actualNode = actualVec[i];
				auto& expectedNode = expectedVec[i];

				if(actualNode.distance != expectedNode.distance || actualNode.idx != expectedNode.idx)
					fail(actualRes, algoIdx);
			}
		}
	}

	void DebugHNSWRunner::selectOriginalNeighbors(size_t lc) {
		for(auto& a : this->algos)
			a->getDebugObject()->selectOriginalNeighbors(lc);
	}

	NodeVecPtr DebugHNSWRunner::testOriginalNeighbors(size_t lc) {
		auto expectedRes = this->debugObj()->getOriginalNeighbors();
		auto& expectedVec = *expectedRes;
		auto expectedLen = expectedVec.size();
		auto algosLen = this->len();

		auto fail = [&](const NodeVecPtr& actualRes, size_t algoIdx) {
			{
				std::ofstream stream(this->reportLogPath());
				this->writeReport(TEST_SELECT_NEIGHBORS, algoIdx, stream, true, lc);
			}
			{
				std::ofstream stream(this->expectedLogPath());
				writeNodeVec(expectedRes, stream);
			}
			{
				std::ofstream stream(this->actualLogPath());
				writeNodeVec(actualRes, stream);
			}
			this->throwAppError(TEST_SELECT_NEIGHBORS, algoIdx);
		};

		for(size_t algoIdx = 1; algoIdx < algosLen; algoIdx++) {
			auto actualRes = this->debugObj(algoIdx)->getOriginalNeighbors();
			auto& actualVec = *actualRes;
			auto actualLen = actualVec.size();

			if(actualLen != expectedLen)
				fail(actualRes, algoIdx);

			for(size_t i = 0; i < expectedLen; i++) {
				auto& actualNode = actualVec[i];
				auto& expectedNode = expectedVec[i];

				if(actualNode.distance != expectedNode.distance || actualNode.idx != expectedNode.idx)
					fail(actualRes, algoIdx);
			}
		}

		return expectedRes;
	}

	void DebugHNSWRunner::connect(size_t lc) {
		for(auto& a : this->algos)
			a->getDebugObject()->connect(lc);
	}

	void DebugHNSWRunner::testConnections(size_t nodeIdx, size_t lc) {
		auto algosLen = this->len();
		auto expectedNeighbors = this->debugObj()->getNeighborsForNode(nodeIdx, lc);
		auto& expectedVec = *expectedNeighbors;
		const auto expectedLen = expectedVec.size();

		auto fail = [&](const IdxVecPtr& actualNeighbors, size_t algoIdx) {
			{
				std::ofstream stream(this->reportLogPath());
				this->writeReport(TEST_CONNECTIONS, algoIdx, stream, true, lc);
			}
			{
				std::ofstream stream(this->expectedLogPath());
				writeIdxVec(expectedNeighbors, stream);
			}
			{
				std::ofstream stream(this->actualLogPath());
				writeIdxVec(actualNeighbors, stream);
			}
			this->throwAppError(TEST_CONNECTIONS, algoIdx);
		};

		for(size_t algoIdx = 1; algoIdx < algosLen; algoIdx++) {
			auto actualNeighbors = this->debugObj(algoIdx)->getNeighborsForNode(nodeIdx, lc);
			auto& actualVec = *actualNeighbors;

			if(actualVec.size() != expectedLen)
				fail(actualNeighbors, algoIdx);

			for(size_t i = 0; i < expectedLen; i++)
				if(actualVec[i] != expectedVec[i])
					fail(actualNeighbors, algoIdx);
		}
	}

	void DebugHNSWRunner::testConnections(const NodeVecPtr& neighbors, size_t queryIdx, size_t lc) {
		this->testConnections(queryIdx, lc);

		for(auto& n : *neighbors)
			this->testConnections(n.idx, lc);
	}

	void DebugHNSWRunner::prepareNextLayer(size_t lc) {
		for(auto& a : this->algos)
			a->getDebugObject()->prepareNextLayer(lc);
	}

	void DebugHNSWRunner::setupEnterPoint() {
		for(auto& a : this->algos)
			a->getDebugObject()->setupEnterPoint();
	}

	void DebugHNSWRunner::testEnterPoint() {
		auto expectedEntry = this->debugObj()->getEnterPoint();
		auto len = this->len();

		for(size_t i = 1; i < len; i++) {
			auto actualEntry = this->debugObj(i)->getEnterPoint();

			if(actualEntry != expectedEntry) {
				{
					std::ofstream stream(this->reportLogPath());
					this->writeReport(TEST_ENTRY, i, stream);
				}
				{
					std::ofstream stream(this->expectedLogPath());
					stream << "Entry index: " << expectedEntry << '\n';
				}
				{
					std::ofstream stream(this->actualLogPath());
					stream << "Entry index: " << actualEntry << '\n';
				}
				this->throwAppError(TEST_ENTRY, i);
			}
		}
	}

	void DebugHNSWRunner::build(const FloatVecPtr& coords, size_t dim) {
		for(auto& a : this->algos)
			a->init();

		auto& c = *coords;
		const auto count = coords->size() / dim;

		for(size_t i = 0; i < count; i++)
			this->insert(&c[i * dim], i);
	}

	DebugHNSWRunner::DebugHNSWRunner(const HNSWAlgoVec& algos, const fs::path& outDir) : algos(algos), currentIdx(0), outDir(outDir) {}
}
