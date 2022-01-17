#pragma once
#include <random>
#include <unordered_map>
#include "KNNAlgo.hpp"

namespace chm {
	struct Config {
		size_t dim;

		size_t efConstruction;
		size_t M;
		float mL;
		size_t Mmax;
		size_t Mmax0;

		bool useHeuristic;
		bool extendCandidates;
		bool keepPrunedConnections;

		void calcML();
		Config(const HNSWConfigPtr& cfg);
		Config(size_t dim);
		Config& setHeuristic(bool extendCandidates = false, bool keepPrunedConnections = false);
	};

	struct NodeDistance {
		float distance;
		size_t nodeID;
	};

	typedef std::vector<NodeDistance> NodeDistanceVec;

	struct FurthestComparator {
		constexpr bool operator()(const NodeDistance& a, const NodeDistance& b) const noexcept;
	};

	class FurthestHeap : public Unique {
		FurthestComparator cmp;

	public:
		NodeDistanceVec nodes;

		void clear();
		FurthestHeap();
		FurthestHeap(NodeDistanceVec& ep);
		void push(float distance, size_t nodeID);
		NodeDistance pop();
		NodeDistance top();
	};

	struct NearestComparator {
		constexpr bool operator()(const NodeDistance& a, const NodeDistance& b) const noexcept;
	};

	class NearestHeap : public Unique {
		NearestComparator cmp;

	public:
		NodeDistanceVec nodes;

		void clear();
		void fillLayer(IdxVec& layer);
		bool isCloserThanAny(NodeDistance& node);
		void keepNearest(size_t K);
		NearestHeap();
		NearestHeap(NearestHeap& other);
		void push(float distance, size_t nodeID);
		NodeDistance pop();
		void remove(size_t nodeID);
		void reserve(size_t s);
		size_t size();
		void swap(NearestHeap& other);
		NodeDistance& top();
	};

	class DynamicList : public Unique {
		void clear();

	public:
		FurthestHeap furthestHeap;
		NearestHeap nearestHeap;

		void add(float distance, size_t nodeID);
		DynamicList(float distance, size_t entryID);
		void fillResults(size_t K, IdxVec& outIDs, FloatVec& outDistances);
		NodeDistance furthest();
		void keepOnlyNearest();
		void removeFurthest();
		size_t size();
	};

	enum class State {
		INSERTING,
		SEARCHING,
		SHRINKING
	};

	class Graph : public Unique {
	public:
		Config cfg;

		size_t entryID;
		size_t entryLevel;

		std::unordered_map<size_t, float> distances;
		IdxVec3D layers;

		float* coords;
		float* queryCoords;

		std::default_random_engine gen;
		std::uniform_real_distribution<float> dist;

		std::ostream* debugStream;
		size_t nodeCount;

		float getDistance(size_t nodeID, size_t queryID, State s = State::INSERTING);
		size_t getNewLevel();

		void insert(size_t queryID);
		void searchLayer(size_t queryID, DynamicList& W, size_t ef, size_t lc, State s = State::INSERTING);
		void selectNeighbors(size_t queryID, NearestHeap& outC, size_t M, size_t lc, State s = State::INSERTING);
		void selectNeighborsHeuristic(size_t queryID, NearestHeap& outC, size_t M, size_t lc, State s);
		void selectNeighborsSimple(NearestHeap& outC, size_t M);
		void knnSearch(size_t queryID, size_t K, size_t ef, IdxVec& outIDs, FloatVec& outDistances);

		void connect(size_t queryID, NearestHeap& neighbors, size_t lc);
		void fillHeap(size_t queryID, IdxVec& eConn, NearestHeap& eNewConn);
		void initLayers(size_t queryID, size_t level);

		Graph(const Config& cfg, unsigned int seed, bool useRndSeed);

		void build(float* coords, size_t count);
		void search(
			float* queryCoords, size_t queryCount, size_t K, size_t ef,
			IdxVec2D& outIDs, FloatVec2D& outDistances
		);

		size_t getNodeCount();
		void printLayers(std::ostream& s);
		void setDebugStream(std::ostream& s);

		void init(float* coords, size_t count);
	};

	class GraphWrapper : public HNSWAlgo {
		size_t ef;

	protected:
		Graph* hnsw;

		void insert(float* data, size_t idx) override;

	public:
		IdxVec3DPtr getConnections() const override;
		DebugHNSW* getDebugObject() override;
		GraphWrapper(const HNSWConfigPtr& cfg);
		void init() override;
		void setSearchEF(size_t ef) override;
	};
}
