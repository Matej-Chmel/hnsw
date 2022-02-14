#pragma once
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "distances.hpp"
#include "Heap.hpp"

namespace chm {
	template<typename Dist>
	class HnswOrig {
		std::vector<std::vector<IdxVec>> connections;
		std::vector<Dist> coords;
		const size_t dim;
		const DistFunc<Dist> distFunc;
		std::unordered_map<Idx, Dist> distanceCache;
		const size_t efConstruction;
		Idx entryIdx;
		size_t entryLevel;
		std::default_random_engine gen;
		const size_t M;
		const double mL;
		const size_t Mmax0;
		Idx nodeCount;

		void fillHeap(const Dist* const query, const Dist* const insertedQuery, IdxVec& eConn, FarHeap<Dist>& eNewConn);
		const Dist* const getCoords(const Idx idx) const;
		Dist getDistance(const Dist* const node, const Dist* const query, const bool useCache = false, const Idx nodeIdx = 0);
		IdxVec& getNeighbors(const Idx idx, const size_t lc);
		size_t getNewLevel();
		void initConnections(const Idx idx, const size_t level);
		void searchLowerLayer(const Dist* const query, Node<Dist>& ep, const size_t ef, const size_t lc, FarHeap<Dist>& W);
		void searchUpperLayer(const Dist* const query, Node<Dist>& resEp, const size_t lc);
		void selectNeighborsHeuristic(FarHeap<Dist>& outC, const size_t M);

	public:
		HnswOrig(
			const size_t dim, const DistFunc<Dist> distFunc, const size_t efConstruction,
			const size_t M, const size_t maxNodeCount, const Idx seed
		);
		void insert(const Dist* const query);
		FarHeap<Dist> search(const Dist* const query, const size_t K, const size_t ef);
	};

	template<typename Dist>
	inline void HnswOrig<Dist>::fillHeap(const Dist* const query, const Dist* const insertedQuery, IdxVec& eConn, FarHeap<Dist>& eNewConn) {
		eNewConn.clear();
		eNewConn.reserve(eConn.size() + 1);
		eNewConn.push(this->getDistance(insertedQuery, query), this->nodeCount);

		for(const auto& idx : eConn)
			eNewConn.push(this->getDistance(this->getCoords(idx), query), idx);
	}

	template<typename Dist>
	inline const Dist* const HnswOrig<Dist>::getCoords(const Idx idx) const {
		return this->coords.data() + idx * this->dim;
	}

	template<typename Dist>
	inline Dist HnswOrig<Dist>::getDistance(const Dist* const node, const Dist* const query, const bool useCache, const Idx nodeIdx) {
		if(useCache) {
			const auto iter = this->distanceCache.find(nodeIdx);

			if(iter != this->distanceCache.end())
				return iter->second;
		}

		const auto res = this->distFunc(node, query, this->dim);

		if(useCache)
			this->distanceCache[nodeIdx] = res;

		return res;
	}

	template<typename Dist>
	inline IdxVec& HnswOrig<Dist>::getNeighbors(const Idx idx, const size_t lc) {
		return this->connections[idx][lc];
	}

	template<typename Dist>
	inline size_t HnswOrig<Dist>::getNewLevel() {
		std::uniform_real_distribution<double> dist(0.0, 1.0);
		return size_t(-std::log(dist(this->gen)) * this->mL);
	}

	template<typename Dist>
	inline void HnswOrig<Dist>::initConnections(const Idx idx, const size_t level) {
		this->connections[idx].resize(level + 1);
	}

	template<typename Dist>
	inline void HnswOrig<Dist>::searchLowerLayer(const Dist* const query, Node<Dist>& ep, const size_t ef, const size_t lc, FarHeap<Dist>& W) {
		NearHeap<Dist> C{ep};
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

	template<typename Dist>
	inline void HnswOrig<Dist>::searchUpperLayer(const Dist* const query, Node<Dist>& resEp, const size_t lc) {
		Idx prevIdx{};

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

	template<typename Dist>
	inline void HnswOrig<Dist>::selectNeighborsHeuristic(FarHeap<Dist>& outC, const size_t M) {
		if(outC.len() < M)
			return;

		auto& R = outC;
		NearHeap<Dist> W(outC);

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

	template<typename Dist>
	inline HnswOrig<Dist>::HnswOrig(
		const size_t dim, const DistFunc<Dist> distFunc, const size_t efConstruction,
		const size_t M, const size_t maxNodeCount, const Idx seed
	) : dim(dim), distFunc(distFunc), efConstruction(efConstruction), entryIdx(0), entryLevel(0),
		M(M), mL(1.0 / std::log(1.0 * this->M)), Mmax0(this->M * 2), nodeCount(0) {

		this->connections.resize(maxNodeCount);
		this->coords.resize(this->dim * maxNodeCount);
		this->gen.seed(seed);
	}

	template<typename Dist>
	inline void HnswOrig<Dist>::insert(const Dist* const query) {
		std::copy(query, query + this->dim, this->coords.begin() + this->nodeCount * this->dim);

		if(!this->nodeCount) {
			this->entryLevel = this->getNewLevel();
			this->nodeCount = 1;
			this->initConnections(this->entryIdx, this->entryLevel);
			return;
		}

		this->distanceCache.clear();

		Node<Dist> ep(this->getDistance(this->getCoords(this->entryIdx), query, true, this->entryIdx), this->entryIdx);
		const auto L = this->entryLevel;
		const auto l = this->getNewLevel();

		this->initConnections(this->nodeCount, l);

		for(auto lc = L; lc > l; lc--)
			this->searchUpperLayer(query, ep, lc);

		for(auto lc = std::min(L, l);; lc--) {
			FarHeap<Dist> candidates{};
			this->searchLowerLayer(query, ep, this->efConstruction, lc, candidates);
			this->selectNeighborsHeuristic(candidates, this->M);

			auto& neighbors = this->getNeighbors(this->nodeCount, lc);
			candidates.extractTo(neighbors);

			// ep = nearest from candidates
			ep = Node<Dist>(candidates.top());
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

	template<typename Dist>
	inline FarHeap<Dist> HnswOrig<Dist>::search(const Dist* const query, const size_t K, const size_t ef) {
		this->distanceCache.clear();

		Node<Dist> ep(this->getDistance(this->getCoords(this->entryIdx), query, true, this->entryIdx), this->entryIdx);
		const auto L = this->entryLevel;

		for(size_t lc = L; lc > 0; lc--)
			this->searchUpperLayer(query, ep, lc);

		FarHeap<Dist> W{};
		this->searchLowerLayer(query, ep, ef, 0, W);

		while(W.len() > K)
			W.pop();

		return W;
	}
}
