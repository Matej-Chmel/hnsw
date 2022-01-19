#pragma once
#include <random>
#include <unordered_map>
#include "KNNAlgo.hpp"

namespace chm {
	struct Config {
		size_t dim;
		size_t maxNodeCount;

		size_t efConstruction;
		size_t M;
		double mL;
		size_t Mmax;
		size_t Mmax0;

		bool useHeuristic;
		bool extendCandidates;
		bool keepPrunedConnections;

		void calcML();
		Config(const HNSWConfigPtr& cfg);
	};

	struct FurthestComparator {
		constexpr bool operator()(const Node& a, const Node& b) const noexcept;
	};

	class FurthestHeap : public Unique {
		FurthestComparator cmp;

	public:
		NodeVec nodes;

		void clear();
		FurthestHeap();
		FurthestHeap(NodeVec& ep);
		void push(float distance, size_t nodeID);
		Node pop();
		Node top();
	};

	struct NearestComparator {
		constexpr bool operator()(const Node& a, const Node& b) const noexcept;
	};

	class NearestHeap : public Unique {
		NearestComparator cmp;

	public:
		NodeVec nodes;

		void clear();
		void fillLayer(IdxVec& layer);
		void keepNearest(size_t K);
		NearestHeap();
		NearestHeap(NearestHeap& other);
		void push(float distance, size_t nodeID);
		Node pop();
		void remove(size_t nodeID);
		void reserve(size_t s);
		size_t size();
		void swap(NearestHeap& other);
		Node& top();
	};

	class DynamicList : public Unique {
	public:
		FurthestHeap furthestHeap;
		NearestHeap nearestHeap;

		void add(float distance, size_t nodeID);
		void clear();
		DynamicList() = default;
		DynamicList(float distance, size_t entryID);
		void fillResults(size_t K, IdxVec& outIDs, FloatVec& outDistances);
		Node furthest();
		void keepOnlyNearest();
		void removeFurthest();
		size_t size();
	};

	class Graph : public Unique {
	public:
		Config cfg;

		size_t entryID;
		size_t entryLevel;

		std::unordered_map<size_t, float> distancesCache;
		IdxVec3D layers;

		FloatVec coords;

		std::default_random_engine gen;

		std::ostream* debugStream;
		size_t nodeCount;

		float getDistance(const float* node, const float* query, bool useCache = false, size_t nodeID = 0);
		size_t getNewLevel();

		void insert(const float* coords, size_t queryID);
		void searchLayer(const float* query, DynamicList& W, size_t ef, size_t lc);
		void selectNeighbors(const float* query, NearestHeap& outC, size_t M, size_t lc, bool useCache);
		void selectNeighborsHeuristic(const float* query, NearestHeap& outC, size_t M, size_t lc, bool useCache);
		void selectNeighborsSimple(NearestHeap& outC, size_t M);
		void knnSearch(const float* query, size_t K, size_t ef, IdxVec& outIDs, FloatVec& outDistances);

		void connect(size_t queryID, NearestHeap& neighbors, size_t lc);
		void fillHeap(const float* query, IdxVec& eConn, NearestHeap& eNewConn);
		void initLayers(size_t queryID, size_t level);

		Graph(const Config& cfg, unsigned int seed);

		void build(const FloatVec& coords);
		void search(const FloatVec& queryCoords, size_t K, size_t ef, IdxVec2D& outIDs, FloatVec2D& outDistances);

		size_t getNodeCount();
		void printLayers(std::ostream& s);
		void setDebugStream(std::ostream& s);

		const float* getCoords(size_t idx);
	};

	class GraphWrapper : public HNSWAlgo {
		size_t ef;

	protected:
		Graph* hnsw;

		void insert(float* data, size_t idx) override;

	public:
		virtual ~GraphWrapper();
		IdxVec3DPtr getConnections() const override;
		DebugHNSW* getDebugObject() override;
		GraphWrapper(const HNSWConfigPtr& cfg);
		GraphWrapper(const HNSWConfigPtr& cfg, const std::string& name);
		void init() override;
		KNNResultPtr search(const FloatVecPtr& coords, size_t K) override;
		void setSearchEF(size_t ef) override;
	};

	struct GraphLocals {
		float* coords;
		size_t L;
		size_t l;
		size_t layerMmax;
		size_t queryID;
		bool isFirstNode;
		NearestHeap neighbors;
		DynamicList W;
	};

	class DebugGraph : public DebugHNSW {
		Graph* hnsw;
		GraphLocals local;

	public:
		DebugGraph(Graph* hnsw);

		void startInsert(float* coords, size_t idx);
		size_t getLatestLevel();
		void prepareUpperSearch();
		LevelRange getUpperRange();
		void searchUpperLayers(size_t lc);
		Node getNearestNode();
		void prepareLowerSearch();
		LevelRange getLowerRange();
		void searchLowerLayers(size_t lc);
		NodeVecPtr getLowerLayerResults();
		void selectOriginalNeighbors(size_t lc);
		NodeVecPtr getOriginalNeighbors();
		void connect(size_t lc);
		IdxVecPtr getNeighborsForNode(size_t nodeIdx, size_t lc);
		void prepareNextLayer(size_t lc);
		void setupEnterPoint();
		size_t getEnterPoint();
	};

	class GraphDebugWrapper : public GraphWrapper {
		DebugGraph* debugObj;

	protected:
		void insert(float* data, size_t idx) override;

	public:
		~GraphDebugWrapper();
		GraphDebugWrapper(const HNSWConfigPtr& cfg);
		DebugHNSW* getDebugObject() override;
		void init() override;
	};
}
