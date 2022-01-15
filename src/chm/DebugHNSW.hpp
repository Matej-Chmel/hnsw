#pragma once
#include "KNNAlgo.hpp"

namespace chm {
	struct DebugNode {
		float distance;
		size_t idx;
	};

	typedef std::vector<DebugNode> DebugNodeVec;
	typedef std::shared_ptr<DebugNodeVec> DebugNodeVecPtr;

	class DebugHNSW {
	public:
		virtual ~DebugHNSW() = default;
		virtual void startInsert(float* coords, size_t idx) = 0;
		virtual size_t getLatestLevel() = 0;
		virtual void prepareLowerSearch() = 0;
		virtual void searchUpperLayers(size_t lc) = 0;
		virtual DebugNode getNearestNode() = 0;
		virtual void prepareUpperSearch() = 0;
		virtual void searchLowerLayers(size_t lc) = 0;
		virtual DebugNodeVecPtr getLowerLayerResults() = 0;
		virtual void selectOriginalNeighbors(size_t lc) = 0;
		virtual DebugNodeVecPtr getOriginalNeighbors(size_t lc) = 0;
		virtual void connect(size_t lc) = 0;
		virtual IdxVecPtr getNeighborsForNode(size_t nodeIdx, size_t lc) = 0;
		virtual void prepareNextLayer(size_t lc) = 0;
		virtual void setupEnterPoint() = 0;
		virtual size_t getEnterPoint() = 0;
	};
}
