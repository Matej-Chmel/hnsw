#include "DebugHNSWRunner.hpp"

namespace chm {
	auto TEST_LEVEL = "Last generated level";
	auto TEST_NEAREST_NODE = "Nearest node after upper search";
	auto TEST_LOWER_SEARCH = "Search of lower layers";
	auto TEST_SELECT_NEIGHBORS = "Selecting neighbors";
	auto TEST_CONNECTIONS = "Connections";
	auto TEST_ENTRY = "Entry point";

	std::string DebugHNSWRunner::algoName(size_t i) {
		return this->algos[i]->getName();
	}

	DebugHNSW* DebugHNSWRunner::debugObj(size_t i) {
		return this->algos[i]->getDebugObject();
	}

	void DebugHNSWRunner::insert(float* coords, size_t idx) {
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

	size_t DebugHNSWRunner::len() {
		return this->algos.size();
	}

	void DebugHNSWRunner::throwAppError(const char* testName, size_t algoIdx) {
		throw AppError(
			"Algorithm "_f << this->algoName(algoIdx) << " failed test \"" << testName <<
			"\" when inserting node " << this->currentIdx << '.'
		);
	}

	void DebugHNSWRunner::writeHeader(const char* testName, size_t algoIdx, bool useLayer, size_t lc) {
		*this->stream <<
			"Error report\nReference algorithm: " << this->algoName() << "\nWrong algorithm: " << this->algoName(algoIdx) <<
			"\nFailed test: " << testName;

		if(useLayer)
			*this->stream << "\nLayer: " << lc;

		*this->stream << "\n\n";
	}

	void DebugHNSWRunner::writeIdxVec(const char* name, const IdxVecPtr& vec) {
		auto& v = *vec;
		const auto len = v.size();

		*this->stream << name << " :\n[Length " << len << "]\n\n";

		for(size_t i = 0; i < len; i++)
			*this->stream << "[" << i << "]: " << v[i] << '\n';
	}

	void DebugHNSWRunner::writeNodeVec(const char* name, const NodeVecPtr& vec) {
		auto& v = *vec;
		const auto len = v.size();

		*this->stream << name << " :\n[Length " << len << "]\n\n";

		for(size_t i = 0; i < len; i++) {
			auto& node = v[i];
			*this->stream << "[" << i << "]: " << node.idx << ", " << node.distance << '\n';
		}
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
				this->writeHeader(TEST_LEVEL, i);
				*this->stream << "Expected level: " << expectedLevel << "\nActual level: " << actualLevel << '\n';
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
				this->writeHeader(TEST_NEAREST_NODE, i, true, lc);
				*this->stream <<
					"Expected distance: " << expectedNode.distance << "\nActual distance: " << actualNode.distance <<
					"\nExpected index: " << expectedNode.idx << "\nActual index: " << actualNode.idx << '\n';
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
			this->writeHeader(TEST_LOWER_SEARCH, algoIdx, true, lc);
			this->writeNodeVec("Expected nodes", expectedRes);
			this->writeNodeVec("\nActual nodes", actualRes);
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
			this->writeHeader(TEST_SELECT_NEIGHBORS, algoIdx, true, lc);
			this->writeNodeVec("Expected nodes", expectedRes);
			this->writeNodeVec("\nActual nodes", actualRes);
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
			this->writeHeader(TEST_CONNECTIONS, algoIdx, true, lc);
			this->writeIdxVec("Expected neighbors", expectedNeighbors);
			this->writeIdxVec("\nActual neighbors", actualNeighbors);
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
				this->writeHeader(TEST_ENTRY, i);
				*this->stream << "Expected entry index: " << expectedEntry << "\nActual entry index: " << actualEntry << '\n';
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

	DebugHNSWRunner::DebugHNSWRunner(const HNSWAlgoVec& algos, std::ostream& stream) : algos(algos), currentIdx(0), stream(&stream) {}
}
