#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace chm {
	template<typename Coord, typename Idx>
	struct Node {
		Coord dist;
		Idx idx;

		Node();
		Node(Coord dist, Idx idx);
	};

	template<typename Coord, typename Idx>
	struct FarComparator {
		using Node = Node<Coord, Idx>;

		constexpr bool operator()(const Node& a, const Node& b) const noexcept;
	};

	template<typename Coord, typename Idx>
	struct NearComparator {
		using Node = Node<Coord, Idx>;

		constexpr bool operator()(const Node& a, const Node& b) const noexcept;
	};

	class Unique {
	protected:
		Unique() = default;

	public:
		Unique& operator=(const Unique&) = delete;
		Unique& operator=(Unique&&) = delete;
		Unique(const Unique&) = delete;
		Unique(Unique&&) = delete;
	};

	template<typename Coord, typename Idx, class Comparator>
	class Heap : public Unique {
	public:
		using Node = Node<Coord, Idx>;
		using IdxVec = std::vector<Idx>;

	private:
		using NodeVec = std::vector<Node>;
		typedef typename NodeVec::iterator iterator;

		NodeVec nodes;

		iterator begin();
		iterator end();

	public:
		typedef typename NodeVec::const_iterator const_iterator;

		const_iterator begin() const noexcept;
		void clear();
		const_iterator end() const noexcept;
		void extractTo(IdxVec& v);
		Heap() = default;
		Heap(const Node& ep);
		template<class C> Heap(Heap<Coord, Idx, C>& o);
		size_t len() const;
		void pop();
		void push(const Node& n);
		void push(const Coord dist, const Idx idx);
		void reserve(size_t capacity);
		const Node& top();
	};

	template<typename Coord, typename Idx>
	using FarHeap = Heap<Coord, Idx, FarComparator<Coord, Idx>>;

	template<typename Coord, typename Idx>
	using NearHeap = Heap<Coord, Idx, NearComparator<Coord, Idx>>;

	using IdxVec = std::vector<size_t>;
	using IdxVec3D = std::vector<std::vector<IdxVec>>;
	using IdxVec3DPtr = std::shared_ptr<IdxVec3D>;

	template<typename Coord>
	using ConstIter = typename std::vector<Coord>::const_iterator;

	template<typename Coord>
	struct AbstractHnsw : public Unique {
		using CoordVec = std::vector<Coord>;
		using CoordConstIter = ConstIter<Coord>;

		virtual ~AbstractHnsw() = default;
		virtual IdxVec3DPtr getConnections() const = 0;
		virtual void insert(const CoordConstIter& query) = 0;
		virtual void knnSearch(const CoordConstIter& query, const size_t K, const size_t ef, IdxVec& resIndices, CoordVec& resDistances) = 0;
	};

	template<typename Coord>
	using AbstractHnswPtr = std::shared_ptr<AbstractHnsw<Coord>>;

	struct HnswConfig : public Unique {
		const size_t dim;
		const size_t efConstruction;
		const size_t M;
		const size_t maxNodeCount;
		const unsigned int seed;
		const bool useEuclid;

		HnswConfig(
			const size_t dim, const size_t efConstruction, const size_t M, const size_t maxNodeCount, const unsigned int seed, const bool useEuclid
		);
	};

	using HnswConfigPtr = std::shared_ptr<HnswConfig>;

	#ifdef CHM_HNSW_INTERMEDIATE

	template<typename Coord, typename Idx, bool useEuclid>
	class HnswIntermediate;

	#endif

	template<typename Coord, typename Idx, bool useEuclid>
	class Hnsw : public AbstractHnsw<Coord> {
		#ifdef CHM_HNSW_INTERMEDIATE

		friend HnswIntermediate;

		#endif

		using CoordVec = typename AbstractHnsw<Coord>::CoordVec;
		using CoordConstIter = typename AbstractHnsw<Coord>::CoordConstIter;
		using FarHeap = FarHeap<Coord, Idx>;
		using IdxVec = std::vector<Idx>;
		using NearHeap = NearHeap<Coord, Idx>;
		using Node = Node<Coord, Idx>;

		CoordVec coords;
		std::vector<std::vector<IdxVec>> connections;
		size_t dim;
		std::unordered_map<Idx, Coord> distancesCache;
		size_t efConstruction;
		Idx entryIdx;
		size_t entryLevel;
		std::default_random_engine gen;
		size_t M;
		double mL;
		size_t Mmax0;
		Idx nodeCount;

		void fillHeap(const CoordConstIter& query, const CoordConstIter& insertedQuery, IdxVec& eConn, FarHeap& eNewConn);
		CoordConstIter getCoords(const Idx idx) const;
		Coord getDistance(const CoordConstIter& node, const CoordConstIter& query, const bool useCache = false, const Idx nodeIdx = 0);
		IdxVec& getNeighbors(const Idx idx, const size_t lc);
		size_t getNewLevel();
		void initConnections(const Idx idx, const size_t level);
		void searchLowerLayer(const CoordConstIter& query, Node& ep, const size_t ef, const size_t lc, FarHeap& W);
		void searchUpperLayer(const CoordConstIter& query, Node& resEp, const size_t lc);
		void selectNeighborsHeuristic(FarHeap& outC, const size_t M);

	public:
		IdxVec3DPtr getConnections() const override;
		Hnsw(const HnswConfigPtr& cfg);
		void insert(const CoordConstIter& query) override;
		void knnSearch(const CoordConstIter& query, const size_t K, const size_t ef, chm::IdxVec& resIndices, CoordVec& resDistances) override;
	};

	template<typename Coord>
	AbstractHnswPtr<Coord> createHnsw(const HnswConfigPtr& cfg);

	template<typename Coord, bool useEuclid>
	AbstractHnswPtr<Coord> createHnsw(const HnswConfigPtr& cfg);

	template<typename Coord>
	Coord euclideanDistance(const ConstIter<Coord>& node, const ConstIter<Coord>& query, const size_t dim);

	template<typename Coord>
	Coord innerProductDistance(const ConstIter<Coord>& node, const ConstIter<Coord>& query, const size_t dim);

	template<typename Idx>
	bool isWideEnough(const size_t max);

	template<typename Coord, typename Idx>
	inline Node<Coord, Idx>::Node() : dist(0), idx(0) {}

	template<typename Coord, typename Idx>
	inline Node<Coord, Idx>::Node(Coord dist, Idx idx) : dist(dist), idx(idx) {}

	template<typename Coord, typename Idx>
	inline constexpr bool FarComparator<Coord, Idx>::operator()(const Node& a, const Node& b) const noexcept {
		return a.dist < b.dist;
	}

	template<typename Coord, typename Idx>
	inline constexpr bool NearComparator<Coord, Idx>::operator()(const Node& a, const Node& b) const noexcept {
		return a.dist > b.dist;
	}

	template<typename Coord, typename Idx, class Comparator>
	inline typename Heap<Coord, Idx, Comparator>::iterator Heap<Coord, Idx, Comparator>::begin() {
		return this->nodes.begin();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline typename Heap<Coord, Idx, Comparator>::iterator Heap<Coord, Idx, Comparator>::end() {
		return this->nodes.end();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline typename Heap<Coord, Idx, Comparator>::const_iterator Heap<Coord, Idx, Comparator>::begin() const noexcept {
		return this->nodes.begin();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::clear() {
		this->nodes.clear();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline typename Heap<Coord, Idx, Comparator>::const_iterator Heap<Coord, Idx, Comparator>::end() const noexcept {
		return this->nodes.end();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::extractTo(IdxVec& v) {
		v.clear();
		v.reserve(this->len());

		while(this->len() > 1) {
			v.emplace_back(this->top().idx);
			this->pop();
		}

		v.emplace_back(this->top().idx);
	}

	template<typename Coord, typename Idx, class Comparator>
	inline Heap<Coord, Idx, Comparator>::Heap(const Node& ep) {
		this->push(ep);
	}

	template<typename Coord, typename Idx, class Comparator>
	template<class C>
	inline Heap<Coord, Idx, Comparator>::Heap(Heap<Coord, Idx, C>& o) {
		this->reserve(o.len());

		while(o.len()) {
			this->push(o.top());
			o.pop();
		}
	}

	template<typename Coord, typename Idx, class Comparator>
	inline size_t Heap<Coord, Idx, Comparator>::len() const {
		return this->nodes.size();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::pop() {
		std::pop_heap(this->begin(), this->end(), Comparator());
		this->nodes.pop_back();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::push(const Node& n) {
		this->push(n.dist, n.idx);
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::push(const Coord dist, const Idx idx) {
		this->nodes.emplace_back(dist, idx);
		std::push_heap(this->begin(), this->end(), Comparator());
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::reserve(size_t capacity) {
		this->nodes.reserve(capacity);
	}

	template<typename Coord, typename Idx, class Comparator>
	inline const typename Heap<Coord, Idx, Comparator>::Node& Heap<Coord, Idx, Comparator>::top() {
		return this->nodes.front();
	}

	HnswConfig::HnswConfig(
		const size_t dim, const size_t efConstruction, const size_t M, const size_t maxNodeCount, const unsigned int seed, const bool useEuclid
	) : dim(dim), efConstruction(efConstruction), M(M), maxNodeCount(maxNodeCount), seed(seed), useEuclid(useEuclid) {}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::fillHeap(
		const CoordConstIter& query, const CoordConstIter& insertedQuery, IdxVec& eConn, FarHeap& eNewConn
	) {
		eNewConn.clear();
		eNewConn.reserve(eConn.size());
		eNewConn.push(this->getDistance(insertedQuery, query), this->nodeCount);

		for(const auto& idx : eConn)
			eNewConn.push(this->getDistance(this->getCoords(idx), query), idx);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline typename Hnsw<Coord, Idx, useEuclid>::CoordConstIter Hnsw<Coord, Idx, useEuclid>::getCoords(const Idx idx) const {
		return this->coords.cbegin() + idx * this->dim;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline Coord Hnsw<Coord, Idx, useEuclid>::getDistance(
		const CoordConstIter& node, const CoordConstIter& query, const bool useCache, const Idx nodeIdx
	) {
		if(useCache) {
			auto iter = this->distancesCache.find(nodeIdx);

			if(iter != this->distancesCache.end())
				return iter->second;
		}

		Coord res{};

		if constexpr(useEuclid)
			res = euclideanDistance<Coord>(node, query, this->dim);
		else
			res = innerProductDistance<Coord>(node, query, this->dim);

		if(useCache)
			this->distancesCache[nodeIdx] = res;

		return res;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline typename Hnsw<Coord, Idx, useEuclid>::IdxVec& Hnsw<Coord, Idx, useEuclid>::getNeighbors(const Idx idx, const size_t lc) {
		return this->connections[idx][lc];
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline size_t Hnsw<Coord, Idx, useEuclid>::getNewLevel() {
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		return size_t(
			-std::log(dist(this->gen)) * this->mL
		);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::initConnections(const Idx idx, const size_t level) {
		this->connections[idx].resize(level + 1);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::searchLowerLayer(const CoordConstIter& query, Node& ep, const size_t ef, const size_t lc, FarHeap& W) {
		NearHeap C{ep};
		std::unordered_set<Idx> v{ep.idx};
		W.push(ep);

		while(C.len()) {
			Idx cIdx{};

			{
				const auto& c = C.top();
				const auto& f = W.top();

				if(c.dist > f.dist && W.len() == ef)
					break;

				cIdx = c.idx;
			}

			const auto& neighbors = this->getNeighbors(cIdx, lc);

			// Extract nearest from C.
			C.pop();

			for(const auto& eIdx : neighbors) {
				if(v.insert(eIdx).second) {
					const auto eDist = this->getDistance(this->getCoords(eIdx), query, true, eIdx);
					bool shouldAdd{};

					{
						const auto& f = W.top();
						shouldAdd = f.dist > eDist || W.len() < ef;
					}

					if(shouldAdd) {
						C.push(eDist, eIdx);
						W.push(eDist, eIdx);

						if(W.len() > ef)
							W.pop();
					}
				}
			}
		}
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::searchUpperLayer(const CoordConstIter& query, Node& resEp, const size_t lc) {
		size_t prevIdx{};

		do {
			const auto& neighbors = this->getNeighbors(resEp.idx, lc);
			prevIdx = resEp.idx;

			for(const auto& cIdx : neighbors) {
				const auto cDist = this->getDistance(this->getCoords(cIdx), query, true, cIdx);

				if(cDist < resEp.dist) {
					resEp.dist = cDist;
					resEp.idx = cIdx;
				}
			}

		} while(resEp.idx != prevIdx);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::selectNeighborsHeuristic(FarHeap& outC, const size_t M) {
		if(outC.len() < M)
			return;

		auto& R = outC;
		NearHeap W(outC);

		R.reserve(std::min(W.len(), M));

		while(W.len() && R.len() < M) {
			{
				const auto& e = W.top();
				const auto eCoords = this->getCoords(e.idx);

				for(const auto& rNode : std::as_const(R))
					if(this->getDistance(eCoords, this->getCoords(rNode.idx)) < e.dist)
						goto isNotCloser;

				R.push(e.dist, e.idx);
			}

			isNotCloser:;

			// Extract nearest from W.
			W.pop();
		}
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline IdxVec3DPtr Hnsw<Coord, Idx, useEuclid>::getConnections() const {
		auto res = std::make_shared<IdxVec3D>();
		auto& r = *res;
		r.resize(size_t(this->nodeCount));

		for(size_t i = 0; i < size_t(this->nodeCount); i++) {
			const auto& nodeLayers = this->connections[i];
			const auto layersLen = nodeLayers.size();
			auto& rLayers = r[i];
			rLayers.resize(layersLen);

			for(size_t lc = 0; lc < layersLen; lc++) {
				const auto& layer = nodeLayers[lc];
				const auto layerLen = layer.size();
				auto& rLayer = rLayers[lc];
				rLayer.reserve(layerLen);

				for(size_t neighborIdx = 0; neighborIdx < layerLen; neighborIdx++)
					rLayer.push_back(size_t(layer[neighborIdx]));
			}
		}

		return res;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline Hnsw<Coord, Idx, useEuclid>::Hnsw(const HnswConfigPtr& cfg)
		: dim(cfg->dim), efConstruction(cfg->efConstruction), entryIdx(0), entryLevel(0),
		M(cfg->M), mL(1.0 / std::log(1.0 * this->M)), Mmax0(this->M * 2), nodeCount(0) {

		this->coords.resize(this->dim * cfg->maxNodeCount);
		this->gen.seed(cfg->seed);
		this->connections.resize(cfg->maxNodeCount);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::insert(const CoordConstIter& query) {
		std::copy(query, query + this->dim, this->coords.begin() + this->nodeCount * this->dim);

		if(!this->nodeCount) {
			this->entryLevel = this->getNewLevel();
			this->nodeCount = 1;
			this->initConnections(this->entryIdx, this->entryLevel);
			return;
		}

		this->distancesCache.clear();

		Node ep(this->getDistance(this->getCoords(this->entryIdx), query, true, this->entryIdx), this->entryIdx);
		const auto L = this->entryLevel;
		const auto l = this->getNewLevel();

		this->initConnections(this->nodeCount, l);

		for(auto lc = L; lc > l; lc--)
			this->searchUpperLayer(query, ep, lc);

		for(auto lc = std::min(L, l);; lc--) {
			FarHeap candidates{};
			this->searchLowerLayer(query, ep, this->efConstruction, lc, candidates);
			this->selectNeighborsHeuristic(candidates, this->M);

			auto& neighbors = this->getNeighbors(this->nodeCount, lc);
			candidates.extractTo(neighbors);

			// ep = nearest from candidates
			ep = Node(candidates.top());
			const auto layerMmax = !lc ? this->Mmax0 : this->M;

			for(const auto& eIdx : neighbors) {
				auto& eConn = this->getNeighbors(eIdx, lc);

				if(eConn.size() < layerMmax)
					eConn.push_back(this->nodeCount);
				else {
					this->fillHeap(this->getCoords(eIdx), query, eConn, candidates);
					this->selectNeighborsHeuristic(candidates, layerMmax);
					candidates.extractTo(eConn);
				}
			}

			if(!lc)
				break;
		}

		if(l > L) {
			this->entryIdx = this->nodeCount;
			this->entryLevel = l;
		}

		this->nodeCount++;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::knnSearch(
		const CoordConstIter& query, const size_t K, const size_t ef, chm::IdxVec& resIndices, CoordVec& resDistances
	) {
		this->distancesCache.clear();

		Node ep(this->getDistance(this->getCoords(this->entryIdx), query, true, this->entryIdx), this->entryIdx);
		const auto L = this->entryLevel;

		for(size_t lc = L; lc > 0; lc--)
			this->searchUpperLayer(query, ep, lc);

		FarHeap W{};
		this->searchLowerLayer(query, ep, Idx(ef), 0, W);

		while(W.len() > K)
			W.pop();

		const auto len = W.len();
		resDistances.clear();
		resDistances.resize(len);
		resIndices.clear();
		resIndices.resize(len);

		for(size_t i = len - 1;; i--) {
			{
				const auto& n = W.top();
				resDistances[i] = n.dist;
				resIndices[i] = size_t(n.idx);
			}
			W.pop();

			if(!i)
				break;
		}
	}

	template<typename Coord>
	AbstractHnswPtr<Coord> createHnsw(const HnswConfigPtr& cfg) {
		if(cfg->useEuclid)
			return createHnsw<Coord, true>(cfg);
		return createHnsw<Coord, false>(cfg);
	}

	template<typename Coord, bool useEuclid>
	AbstractHnswPtr<Coord> createHnsw(const HnswConfigPtr& cfg) {
		if(isWideEnough<unsigned short>(cfg->maxNodeCount))
			return std::make_shared<Hnsw<Coord, unsigned short, useEuclid>>(cfg);
		if(isWideEnough<unsigned int>(cfg->maxNodeCount))
			return std::make_shared<Hnsw<Coord, unsigned int, useEuclid>>(cfg);
		return std::make_shared<Hnsw<Coord, size_t, useEuclid>>(cfg);
	}

	template<typename Coord>
	Coord euclideanDistance(const ConstIter<Coord>& node, const ConstIter<Coord>& query, const size_t dim) {
		Coord res = 0;

		for(size_t i = 0; i < dim; i++) {
			auto diff = *(node + i) - *(query + i);
			res += diff * diff;
		}

		return res;
	}

	template<typename Coord>
	Coord innerProductDistance(const ConstIter<Coord>& node, const ConstIter<Coord>& query, const size_t dim) {
		Coord res = 0;

		for(size_t i = 0; i < dim; i++)
			res += *(node + i) * *(query + i);

		return Coord{1} - res;
	}

	template<typename Idx>
	bool isWideEnough(const size_t max) {
		return max <= std::numeric_limits<Idx>::max();
	}
}
