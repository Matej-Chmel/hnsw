#pragma once
#include <algorithm>
#include <random>
#include <unordered_map>
#include "KNNAlgo.hpp"

namespace chm {
	struct Config {
		size_t dim;
		size_t efConstruction;
		size_t M;
		double mL;
		size_t Mmax;
		size_t Mmax0;

		Config(const HNSWConfigPtr& cfg);
	};

	struct FarComparator {
		constexpr bool operator()(const Node& a, const Node& b) const noexcept;
	};

	struct NearComparator {
		constexpr bool operator()(const Node& a, const Node& b) const noexcept;
	};

	template<class Comparator>
	class Heap {
		NodeVec nodes;

	public:
		typedef typename NodeVec::iterator iterator;

		Node& back();
		iterator begin();
		void clear();
		iterator end();
		Heap() = default;
		Heap(Node& ep);
		template<class C> Heap(Heap<C>& o);
		size_t len();
		void push(Node& n);
		void push(float dist, size_t idx);
		void pop();
		void reserve(size_t capacity);
		Node& top();
	};

	typedef Heap<FarComparator> FarHeap;
	typedef Heap<NearComparator> NearHeap;

	class Graph : public Unique {
	public:
		Config cfg;
		IdxVec3D connections;
		FloatVec coords;
		std::unordered_map<size_t, float> distancesCache;
		size_t entryIdx;
		size_t entryLevel;
		std::default_random_engine gen;
		size_t nodeCount;

		void connect(FarHeap& neighborHeap, IdxVec& resNeighbors);
		void fillHeap(const float* query, size_t newIdx, IdxVec& eConn, FarHeap& eNewConn);
		const float* getCoords(size_t idx);
		float getDistance(const float* node, const float* query, bool useCache = false, size_t nodeIdx = 0);
		IdxVec& getNeighbors(size_t idx, size_t lc);
		size_t getNewLevel();
		Graph(const Config& cfg, size_t maxNodeCount, unsigned int seed);
		void initConnections(size_t queryIdx, size_t level);

		void insert(const float* queryCoords, size_t queryIdx);
		void searchUpperLayer(const float* query, Node& resEp, size_t lc);
		void searchLowerLayer(const float* query, Node& ep, size_t ef, size_t lc, FarHeap& W);
		void selectNeighborsHeuristic(FarHeap& outC, size_t M);
		void knnSearch(const float* query, size_t K, size_t ef, IdxVec& resIndices, FloatVec& resDistances);
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
		FarHeap candidates;
		Node ep;
		size_t L;
		size_t l;
		size_t layerMmax;
		float* queryCoords;
		size_t queryIdx;
		bool isFirstNode;
	};

	class DebugGraph : public DebugHNSW {
		Graph* hnsw;
		GraphLocals local;

	public:
		DebugGraph(Graph* hnsw);

		void startInsert(float* coords, size_t idx) override;
		size_t getLatestLevel() override;
		void prepareUpperSearch() override;
		LevelRange getUpperRange() override;
		void searchUpperLayers(size_t lc) override;
		Node getNearestNode() override;
		void prepareLowerSearch() override;
		LevelRange getLowerRange() override;
		size_t getLowerSearchEntry() override;
		void searchLowerLayers(size_t lc) override;
		NodeVecPtr getLowerLayerResults() override;
		void selectOriginalNeighbors(size_t lc) override;
		NodeVecPtr getOriginalNeighbors() override;
		void connect(size_t lc) override;
		IdxVecPtr getNeighborsForNode(size_t nodeIdx, size_t lc) override;
		void prepareNextLayer(size_t lc) override;
		void setupEnterPoint() override;
		size_t getEnterPoint() override;
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

	template<class Comparator>
	inline Node& Heap<Comparator>::back() {
		return this->nodes.back();
	}

	template<class Comparator>
	inline typename Heap<Comparator>::iterator Heap<Comparator>::begin() {
		return this->nodes.begin();
	}

	template<class Comparator>
	inline void Heap<Comparator>::clear() {
		this->nodes.clear();
	}

	template<class Comparator>
	inline typename Heap<Comparator>::iterator Heap<Comparator>::end() {
		return this->nodes.end();
	}

	template<class Comparator>
	inline Heap<Comparator>::Heap(Node& ep) : nodes{ep} {}

	template<class Comparator>
	template<class C>
	inline Heap<Comparator>::Heap(Heap<C>& o) {
		const auto len = o.len();
		this->reserve(len);

		if(len < 2)
			this->push(o.top());
		else {
			const auto shortLen = len - 2;

			for(size_t i = 0; i < shortLen; i++) {
				this->push(o.top());
				o.pop();
			}

			this->push(o.top());
			this->push(o.back());
		}
	}

	template<class Comparator>
	inline size_t Heap<Comparator>::len() {
		return this->nodes.size();
	}

	template<class Comparator>
	inline void Heap<Comparator>::push(Node& n) {
		this->push(n.dist, n.idx);
	}

	template<class Comparator>
	inline void Heap<Comparator>::push(float dist, size_t idx) {
		this->nodes.emplace_back(dist, idx);
		std::push_heap(this->begin(), this->end(), Comparator());
	}

	template<class Comparator>
	inline void Heap<Comparator>::pop() {
		std::pop_heap(this->begin(), this->end(), Comparator());
		this->nodes.pop_back();
	}

	template<class Comparator>
	inline void Heap<Comparator>::reserve(size_t capacity) {
		this->nodes.reserve(capacity);
	}

	template<class Comparator>
	inline Node& Heap<Comparator>::top() {
		return this->nodes.front();
	}
}
