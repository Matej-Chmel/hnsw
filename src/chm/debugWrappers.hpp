#pragma once
#include "DebugHNSW.hpp"
#include "refImplWrappers.hpp"

namespace chm {
	typedef std::priority_queue<
		std::pair<float, hnswlib::tableint>,
		std::vector<std::pair<float, hnswlib::tableint>>,
		hnswlib::HierarchicalNSW<float>::CompareByFirst
	> PriorityQueue;

	struct HnswlibLocals {
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
		bool shouldUpperSearch;
		PriorityQueue top_candidates;

		void clear();
	};

	class DebugHnswlib : public DebugHNSW {
		hnswlib::HierarchicalNSW<float>* hnsw;
		HnswlibLocals local;

		NodeVecPtr vecFromTopCandidates();

	public:
		DebugHnswlib(hnswlib::HierarchicalNSW<float>* hnsw);

		void startInsert(float* coords, size_t idx) override;
		size_t getLatestLevel() override;
		void prepareUpperSearch() override;
		LevelRange getUpperRange() override;
		void searchUpperLayers(size_t lc) override;
		Node getNearestNode() override;
		void prepareLowerSearch() override;
		LevelRange getLowerRange() override;
		void searchLowerLayers(size_t lc) override;
		NodeVecPtr getLowerLayerResults() override;
		void selectOriginalNeighbors(size_t lc) override;
		NodeVecPtr getOriginalNeighbors(size_t lc) override;
		void connect(size_t lc) override;
		IdxVecPtr getNeighborsForNode(size_t nodeIdx, size_t lc) override;
		void prepareNextLayer(size_t lc) override;
		void setupEnterPoint() override;
		size_t getEnterPoint() override;
	};

	class hnswlibDebugWrapper : public hnswlibWrapper {
		DebugHnswlib* debugObj;

	protected:
		void init() override;
		void insert(float* data, size_t idx) override;

	public:
		~hnswlibDebugWrapper();
		hnswlibDebugWrapper(const HNSWConfigPtr& cfg);
	};

	typedef std::vector<baca::Neighbors> NeighborsVec;

	struct BacaLocals {
		int actualMmax;
		baca::Node* down_node;
		baca::Node* ep;
		baca::pointer_t ep_node_order;
		int32_t L;
		int32_t l;
		baca::Node* new_node;
		baca::Node* prev;
		float* q;
		NeighborsVec R;
		NeighborsVec Roverflow;
		NeighborsVec Woverflow;
	};

	class DebugBaca : public DebugHNSW {
		baca::HNSW* hnsw;
		BacaLocals local;

	public:
		DebugBaca(baca::HNSW* hnsw);

		void startInsert(float* coords, size_t idx) override;
		size_t getLatestLevel() override;
		void prepareUpperSearch() override;
		LevelRange getUpperRange() override;
		void searchUpperLayers(size_t lc) override;
		Node getNearestNode() override;
		void prepareLowerSearch() override;
		LevelRange getLowerRange() override;
		void searchLowerLayers(size_t lc) override;
		NodeVecPtr getLowerLayerResults() override;
		void selectOriginalNeighbors(size_t lc) override;
		NodeVecPtr getOriginalNeighbors(size_t lc) override;
		void connect(size_t lc) override;
		IdxVecPtr getNeighborsForNode(size_t nodeIdx, size_t lc) override;
		void prepareNextLayer(size_t lc) override;
		void setupEnterPoint() override;
		size_t getEnterPoint() override;
	};

	class BacaDebugWrapper : public BacaWrapper {
		DebugBaca* debugObj;

	protected:
		void init() override;
		void insert(float* data, size_t idx) override;

	public:
		~BacaDebugWrapper();
		BacaDebugWrapper(const HNSWConfigPtr& cfg);
	};
}
