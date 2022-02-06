#pragma once
#include <cmath>
#include <limits>
#include <random>
#include <unordered_map>
#include <utility>
#include "IHnsw.hpp"
#include "IVisitedSet.hpp"
#include "Heap.hpp"
#include "hnswDistances.hpp"

namespace chm {
	#ifdef CHM_HNSW_INTERMEDIATE

	template<typename Coord, typename Idx, bool useEuclid>
	class HnswInterImpl;

	#endif

	template<typename Coord, typename Idx, bool useEuclid>
	class Hnsw : public IHnsw<Coord> {
		#ifdef CHM_HNSW_INTERMEDIATE

		friend HnswInterImpl<Coord, Idx, useEuclid>;

		#endif

		using FarHeap = FarHeap<Coord, Idx>;
		using NearHeap = NearHeap<Coord, Idx>;
		using Node = Node<Coord, Idx>;

		std::vector<Coord> coords;
		IConnPtr<Coord, Idx> connections;
		const size_t dim;
		std::unordered_map<Idx, Coord> distanceCache;
		const bool distanceCacheEnabled;
		const size_t efConstruction;
		Idx entryIdx;
		size_t entryLevel;
		std::default_random_engine gen;
		const size_t M;
		const double mL;
		const size_t Mmax0;
		Idx nodeCount;
		IVisitedSetPtr<Idx> visited;

		void fillHeap(const ConstIter<Coord>& query, const ConstIter<Coord>& insertedQuery, FarHeap& eNewConn);
		ConstIter<Coord> getCoords(const Idx idx) const;
		Coord getDistance(const ConstIter<Coord>& node, const ConstIter<Coord>& query, const bool useCache = false, const Idx nodeIdx = 0);
		size_t getNewLevel();
		void searchLowerLayer(const ConstIter<Coord>& query, Node& ep, const size_t ef, const size_t lc, FarHeap& W);
		void searchUpperLayer(const ConstIter<Coord>& query, Node& resEp, const size_t lc);
		void selectNeighborsHeuristic(FarHeap& outC, const size_t M);

	public:
		IdxVec3DPtr getConnections() const override;
		Hnsw(const HnswCfgPtr& cfg, const HnswSettingsPtr& settings);
		void insert(const ConstIter<Coord>& query) override;
		void knnSearch(
			const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
		) override;
	};

	template<typename Coord>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg, const HnswSettingsPtr& settings);

	template<typename Coord, bool useEuclid>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg, const HnswSettingsPtr& settings);

	template<typename Idx>
	bool isWideEnough(const size_t max);

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::fillHeap(
		const ConstIter<Coord>& query, const ConstIter<Coord>& insertedQuery, FarHeap& eNewConn
	) {
		eNewConn.clear();
		eNewConn.reserve(this->connections->len());
		eNewConn.push(this->getDistance(insertedQuery, query), this->nodeCount);

		for(const auto& idx : *this->connections)
			eNewConn.push(this->getDistance(this->getCoords(idx), query), idx);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline ConstIter<Coord> Hnsw<Coord, Idx, useEuclid>::getCoords(const Idx idx) const {
		return this->coords.cbegin() + idx * this->dim;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline Coord Hnsw<Coord, Idx, useEuclid>::getDistance(
		const ConstIter<Coord>& node, const ConstIter<Coord>& query, const bool useCache, const Idx nodeIdx
	) {
		if(this->distanceCacheEnabled && useCache) {
			const auto iter = this->distanceCache.find(nodeIdx);

			if(iter != this->distanceCache.end())
				return iter->second;
		}

		Coord res{};

		if constexpr(useEuclid)
			res = euclideanDistance<Coord>(node, query, this->dim);
		else
			res = innerProductDistance<Coord>(node, query, this->dim);

		if(this->distanceCacheEnabled && useCache)
			this->distanceCache[nodeIdx] = res;

		return res;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline size_t Hnsw<Coord, Idx, useEuclid>::getNewLevel() {
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		return size_t(
			-std::log(dist(this->gen)) * this->mL
		);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::searchLowerLayer(
		const ConstIter<Coord>& query, Node& ep, const size_t ef, const size_t lc, FarHeap& W
	) {
		NearHeap C{ep};
		this->visited->prepare(this->nodeCount, ep.idx);
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

			this->connections->useNeighbors(cIdx, lc);

			// Extract nearest from C.
			C.pop();

			for(const auto& eIdx : *this->connections) {
				if(this->visited->insert(eIdx)) {
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
	inline void Hnsw<Coord, Idx, useEuclid>::searchUpperLayer(const ConstIter<Coord>& query, Node& resEp, const size_t lc) {
		size_t prevIdx{};

		do {
			this->connections->useNeighbors(resEp.idx, lc);
			prevIdx = resEp.idx;

			for(const auto& cIdx : *this->connections) {
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
		return this->connections->getConnections();
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline Hnsw<Coord, Idx, useEuclid>::Hnsw(const HnswCfgPtr& cfg, const HnswSettingsPtr& settings)
		: dim(cfg->dim), distanceCacheEnabled(settings->distanceCacheEnabled), efConstruction(cfg->efConstruction), entryIdx(0), entryLevel(0),
		M(cfg->M), mL(1.0 / std::log(1.0 * this->M)), Mmax0(this->M * 2), nodeCount(0) {

		this->connections = createConn<Coord, Idx>(settings->useConnLayer0, cfg->maxNodeCount, this->M, this->Mmax0);
		this->coords.resize(this->dim * cfg->maxNodeCount);
		this->gen.seed(cfg->seed);
		this->visited = createVisitedSet<Idx>(settings->useBitset);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::insert(const ConstIter<Coord>& query) {
		std::copy(query, query + this->dim, this->coords.begin() + this->nodeCount * this->dim);

		if(!this->nodeCount) {
			this->entryLevel = this->getNewLevel();
			this->nodeCount = 1;
			this->connections->init(this->entryIdx, this->entryLevel);
			return;
		}

		if(this->distanceCacheEnabled)
			this->distanceCache.clear();

		Node ep(this->getDistance(this->getCoords(this->entryIdx), query, true, this->entryIdx), this->entryIdx);
		const auto L = this->entryLevel;
		const auto l = this->getNewLevel();

		this->connections->init(this->nodeCount, l);

		for(auto lc = L; lc > l; lc--)
			this->searchUpperLayer(query, ep, lc);

		for(auto lc = std::min(L, l);; lc--) {
			FarHeap candidates{};
			this->searchLowerLayer(query, ep, this->efConstruction, lc, candidates);
			this->selectNeighborsHeuristic(candidates, this->M);

			this->connections->useNeighbors(this->nodeCount, lc);
			this->connections->fillFrom(candidates);

			// ep = nearest from candidates
			ep = Node(candidates.top());
			const auto layerMmax = !lc ? this->Mmax0 : this->M;

			for(const auto& eIdx : *this->connections) {
				this->connections->useNeighbors(eIdx, lc);

				if(this->connections->len() < layerMmax)
					this->connections->push(this->nodeCount);
				else {
					this->fillHeap(this->getCoords(eIdx), query, candidates);
					this->selectNeighborsHeuristic(candidates, layerMmax);
					this->connections->fillFrom(candidates);
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
		const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
	) {
		if(this->distanceCacheEnabled)
			this->distanceCache.clear();

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

		for(auto i = len - 1;; i--) {
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
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg, const HnswSettingsPtr& settings) {
		if(cfg->useEuclid)
			return createHnsw<Coord, true>(cfg, settings);
		return createHnsw<Coord, false>(cfg, settings);
	}

	template<typename Coord, bool useEuclid>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg, const HnswSettingsPtr& settings) {
		if(isWideEnough<unsigned short>(cfg->maxNodeCount))
			return std::make_shared<Hnsw<Coord, unsigned short, useEuclid>>(cfg, settings);
		if(isWideEnough<unsigned int>(cfg->maxNodeCount))
			return std::make_shared<Hnsw<Coord, unsigned int, useEuclid>>(cfg, settings);
		return std::make_shared<Hnsw<Coord, size_t, useEuclid>>(cfg, settings);
	}

	template<typename Idx>
	bool isWideEnough(const size_t max) {
		return max <= std::numeric_limits<Idx>::max();
	}
}
