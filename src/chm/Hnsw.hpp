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

		std::vector<Coord> coords;
		const size_t dim;
		std::unordered_map<Idx, Coord> distanceCache;
		const bool distanceCacheEnabled;
		const size_t efConstruction;
		Idx entryIdx;
		size_t entryLevel;
		Node<Coord, Idx> ep;
		FarHeap<Coord, Idx> farHeap;
		std::default_random_engine gen;
		const bool keepHeaps;
		const size_t M;
		const double mL;
		const size_t Mmax0;
		NearHeap<Coord, Idx> nearHeap;
		INeighborsPtr<Coord, Idx> neighbors;
		Idx nodeCount;
		IVisitedSetPtr<Idx> visited;

		void fillHeap(const ConstIter<Coord>& query, const ConstIter<Coord>& insertedQuery);
		ConstIter<Coord> getCoords(const Idx idx) const;
		Coord getDistance(const ConstIter<Coord>& node, const ConstIter<Coord>& query, const bool useCache = false, const Idx nodeIdx = 0);
		size_t getNewLevel();
		void reserveHeaps(const size_t ef);
		void resetEp(const ConstIter<Coord>& query);
		void searchLowerLayer(const ConstIter<Coord>& query, const size_t ef, const size_t lc);
		void searchUpperLayer(const ConstIter<Coord>& query, const size_t lc);
		void selectNeighborsHeuristic(const size_t M);
		void shrinkHeaps();

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
	inline void Hnsw<Coord, Idx, useEuclid>::fillHeap(const ConstIter<Coord>& query, const ConstIter<Coord>& insertedQuery) {
		this->farHeap.clear();

		if(!this->keepHeaps)
			this->farHeap.reserve(this->neighbors->len() + 1);

		this->farHeap.push(this->getDistance(insertedQuery, query), this->nodeCount);

		for(const auto& idx : *this->neighbors)
			this->farHeap.push(this->getDistance(this->getCoords(idx), query), idx);
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
	inline void Hnsw<Coord, Idx, useEuclid>::reserveHeaps(const size_t ef) {
		const auto maxLen = std::max(ef, this->Mmax0);
		this->farHeap.reserve(maxLen);
		this->nearHeap.reserve(maxLen);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::resetEp(const ConstIter<Coord>& query) {
		this->ep.dist = this->getDistance(this->getCoords(this->entryIdx), query, true, this->entryIdx);
		this->ep.idx = this->entryIdx;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::searchLowerLayer(const ConstIter<Coord>& query, const size_t ef, const size_t lc) {
		auto& C = this->nearHeap;
		auto& W = this->farHeap;

		C.clear();
		C.push(this->ep);
		this->visited->prepare(this->nodeCount, this->ep.idx);
		W.clear();
		W.push(this->ep);

		while(C.len()) {
			Idx cIdx{};

			{
				const auto& c = C.top();
				const auto& f = W.top();

				if(c.dist > f.dist && W.len() == ef)
					break;

				cIdx = c.idx;
			}

			this->neighbors->use(cIdx, lc);

			// Extract nearest from C.
			C.pop();

			for(const auto& eIdx : *this->neighbors) {
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
	inline void Hnsw<Coord, Idx, useEuclid>::searchUpperLayer(const ConstIter<Coord>& query, const size_t lc) {
		size_t prevIdx{};

		do {
			this->neighbors->use(this->ep.idx, lc);
			prevIdx = this->ep.idx;

			for(const auto& cIdx : *this->neighbors) {
				const auto cDist = this->getDistance(this->getCoords(cIdx), query, true, cIdx);

				if(cDist < this->ep.dist) {
					this->ep.dist = cDist;
					this->ep.idx = cIdx;
				}
			}

		} while(this->ep.idx != prevIdx);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::selectNeighborsHeuristic(const size_t M) {
		if(this->farHeap.len() < M)
			return;

		auto& R = this->farHeap;
		auto& W = this->nearHeap;

		W.loadFrom(R, !this->keepHeaps);

		if(!this->keepHeaps)
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
	inline void Hnsw<Coord, Idx, useEuclid>::shrinkHeaps() {
		this->farHeap.shrinkToFit();
		this->nearHeap.shrinkToFit();
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline IdxVec3DPtr Hnsw<Coord, Idx, useEuclid>::getConnections() const {
		return this->neighbors->getConnections();
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline Hnsw<Coord, Idx, useEuclid>::Hnsw(const HnswCfgPtr& cfg, const HnswSettingsPtr& settings)
		: dim(cfg->dim), distanceCacheEnabled(settings->distanceCacheEnabled), efConstruction(cfg->efConstruction), entryIdx(0), entryLevel(0),
		keepHeaps(settings->keepHeaps), M(cfg->M), mL(1.0 / std::log(1.0 * this->M)), Mmax0(this->M * 2), nodeCount(0) {

		this->coords.resize(this->dim * cfg->maxNodeCount);
		this->gen.seed(cfg->seed);
		this->neighbors = createNeighbors<Coord, Idx>(settings->usePreAllocNeighbors, cfg->maxNodeCount, this->M, this->Mmax0);
		this->visited = createVisitedSet<Idx>(settings->useBitset);

		if(this->keepHeaps)
			this->reserveHeaps(this->efConstruction);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::insert(const ConstIter<Coord>& query) {
		std::copy(query, query + this->dim, this->coords.begin() + this->nodeCount * this->dim);

		if(!this->nodeCount) {
			this->entryLevel = this->getNewLevel();
			this->nodeCount = 1;
			this->neighbors->init(this->entryIdx, this->entryLevel);
			return;
		}

		if(this->distanceCacheEnabled)
			this->distanceCache.clear();

		this->resetEp(query);
		const auto L = this->entryLevel;
		const auto l = this->getNewLevel();

		this->neighbors->init(this->nodeCount, l);

		for(auto lc = L; lc > l; lc--)
			this->searchUpperLayer(query, lc);

		for(auto lc = std::min(L, l);; lc--) {
			this->searchLowerLayer(query, this->efConstruction, lc);
			this->selectNeighborsHeuristic(this->M);

			this->neighbors->use(this->nodeCount, lc);
			this->neighbors->fillFrom(this->farHeap);

			// ep = nearest from candidates
			{
				const auto& n = this->farHeap.top();
				this->ep.dist = n.dist;
				this->ep.idx = n.idx;
			}
			const auto layerMmax = !lc ? this->Mmax0 : this->M;

			for(const auto& eIdx : *this->neighbors) {
				this->neighbors->use(eIdx, lc);

				if(this->neighbors->len() < layerMmax)
					this->neighbors->push(this->nodeCount);
				else {
					this->fillHeap(this->getCoords(eIdx), query);
					this->selectNeighborsHeuristic(layerMmax);
					this->neighbors->fillFrom(this->farHeap);
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

		if(!this->keepHeaps)
			this->shrinkHeaps();
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::knnSearch(
		const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
	) {
		if(this->distanceCacheEnabled)
			this->distanceCache.clear();

		if(this->keepHeaps)
			this->reserveHeaps(ef);

		this->resetEp(query);
		const auto L = this->entryLevel;

		for(size_t lc = L; lc > 0; lc--)
			this->searchUpperLayer(query, lc);

		auto& W = this->farHeap;
		this->searchLowerLayer(query, Idx(ef), 0);

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

		if(!this->keepHeaps)
			this->shrinkHeaps();
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
