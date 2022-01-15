#pragma once
#include "DebugHNSW.hpp"
#include "hnswlib/hnswalg.h"

namespace chm {
	typedef std::priority_queue<
		std::pair<float, hnswlib::tableint>,
		std::vector<std::pair<float, hnswlib::tableint>>,
		hnswlib::HierarchicalNSW<float>::CompareByFirst
	> PriorityQueue;

	struct HnswlibState {
		hnswlib::tableint cur_c;
		float curdist;
		int curlevel;
		hnswlib::tableint currObj;
		const void* data_point;
		hnswlib::tableint enterpoint_copy;
		bool epDeleted;
		bool isFirstElement;
		bool isUpdate;
		int level;
		int maxlevelcopy;
		size_t Mcurmax;
		bool searchLowerLayers;
		PriorityQueue top_candidates;

		~HnswlibState();
		void clear();
	};

	class DebugHnswlib : public DebugHNSW {
		hnswlib::HierarchicalNSW<float>* hnsw;
		HnswlibState local;
		hnswlib::L2Space* space;

		DebugNodeVecPtr vecFromTopCandidates();

	public:
		~DebugHnswlib();
		DebugHnswlib(const HNSWConfigPtr& cfg);

		void startInsert(float* coords, size_t idx) override;
		size_t getLatestLevel() override;
		void prepareUpperSearch() override;
		void searchUpperLayers(size_t lc) override;
		DebugNode getNearestNode() override;
		void prepareLowerSearch() override;
		void searchLowerLayers(size_t lc) override;
		DebugNodeVecPtr getLowerLayerResults() override;
		void selectOriginalNeighbors(size_t lc) override;
		DebugNodeVecPtr getOriginalNeighbors(size_t lc) override;
		void connect(size_t lc) override;
		IdxVecPtr getNeighborsForNode(size_t nodeIdx, size_t lc) override;
		void prepareNextLayer(size_t lc) override;
		void setupEnterPoint() override;
		size_t getEnterPoint() override;
	};
}
