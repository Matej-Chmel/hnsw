#pragma once
#include "dataOps.hpp"

namespace chm {
	struct Node {
		float distance;
		size_t idx;

		Node();
		Node(float distance, size_t idx);
	};

	typedef std::vector<Node> NodeVec;
	typedef std::shared_ptr<NodeVec> NodeVecPtr;

	struct NodeComparator {
		bool operator()(const Node& a, const Node& b) const;
	};

	struct LevelRange {
		size_t start;
		size_t end;
		bool shouldLoop;
	};

	class DebugHNSW {
	public:
		virtual ~DebugHNSW() = default;
		virtual void startInsert(float* coords, size_t idx) = 0;
		virtual size_t getLatestLevel() = 0;
		virtual void prepareUpperSearch() = 0;
		virtual LevelRange getUpperRange() = 0;
		virtual void searchUpperLayers(size_t lc) = 0;
		virtual Node getNearestNode() = 0;
		virtual void prepareLowerSearch() = 0;
		virtual LevelRange getLowerRange() = 0;
		virtual void searchLowerLayers(size_t lc) = 0;
		virtual NodeVecPtr getLowerLayerResults() = 0;
		virtual void selectOriginalNeighbors(size_t lc) = 0;
		virtual NodeVecPtr getOriginalNeighbors() = 0;
		virtual void connect(size_t lc) = 0;
		virtual IdxVecPtr getNeighborsForNode(size_t nodeIdx, size_t lc) = 0;
		virtual void prepareNextLayer(size_t lc) = 0;
		virtual void setupEnterPoint() = 0;
		virtual size_t getEnterPoint() = 0;
	};
}
