#pragma once
#include "KNNAlgo.hpp"

namespace chm {
	typedef std::vector<HNSWAlgoPtr> HNSWAlgoVec;

	class DebugHNSWRunner {
		HNSWAlgoVec algos;
		size_t currentIdx;
		std::ostream* stream;

		std::string algoName(size_t i = 0);
		DebugHNSW* debugObj(size_t i = 0);
		void insert(float* coords, size_t idx);
		size_t len();
		void throwAppError(const char* testName, size_t algoIdx);
		void writeHeader(const char* testName, size_t algoIdx, bool useLayer = false, size_t lc = 0);
		void writeIdxVec(const char* name, const IdxVecPtr& vec);
		void writeNodeVec(const char* name, const NodeVecPtr& vec);

		void startInsert(float* coords, size_t idx);
		void testLatestLevel();
		void prepareUpperSearch();
		LevelRange getUpperRange();
		void searchUpperLayers(size_t lc);
		void testNearestNode(size_t lc);
		void prepareLowerSearch();
		LevelRange getLowerRange();
		void searchLowerLayers(size_t lc);
		void testLowerLayerResults(size_t lc);
		void selectOriginalNeighbors(size_t lc);
		NodeVecPtr testOriginalNeighbors(size_t lc);
		void connect(size_t lc);
		void testConnections(size_t nodeIdx, size_t lc);
		void testConnections(const NodeVecPtr& neighbors, size_t queryIdx, size_t lc);
		void prepareNextLayer(size_t lc);
		void setupEnterPoint();
		void testEnterPoint();

	public:
		void build(const FloatVecPtr& coords, size_t dim);
		DebugHNSWRunner(const HNSWAlgoVec& algos, std::ostream& stream);
	};
}
