#pragma once
#include <cmath>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "Heap.hpp"

namespace chm {
	#ifdef CHM_HNSW_DEBUG

	template<typename Coord, typename Idx, bool useEuclid>
	class HnswDebug;

	#endif

	template<typename Coord, typename Idx, bool useEuclid>
	class Hnsw {
		#ifdef CHM_HNSW_DEBUG

		friend HnswDebug;

		#endif

	public:
		using CoordVec = std::vector<Coord>;
		using IdxVec = std::vector<Idx>;

	private:
		using FarHeap = FarHeap<Coord, Idx>;
		using NearHeap = NearHeap<Coord, Idx>;
		using Node = Node<Coord, Idx>;
		using IdxVec3D = std::vector<std::vector<IdxVec>>;

		std::vector<Coord> coords;
		IdxVec3D connections;
		size_t dim;
		std::unordered_map<Idx, Coord> distancesCache;
		Idx efConstruction;
		Idx entryIdx;
		Idx entryLevel;
		std::default_random_engine gen;
		Idx M;
		double mL;
		Idx Mmax0;
		Idx nodeCount;

		void fillHeap(const Coord* const query, const Idx newIdx, IdxVec& eConn, FarHeap& eNewConn);
		const Coord* const getCoords(const Idx idx) const;
		Coord getDistance(const Coord* const node, const Coord* const query, const bool useCache = false, const Idx nodeIdx = 0);
		IdxVec& getNeighbors(const Idx idx, const Idx lc);
		Idx getNewLevel();
		void initConnections(const Idx idx, const Idx level);
		void searchLowerLayer(const Coord* const query, Node& ep, const Idx ef, const Idx lc, FarHeap& W);
		void searchUpperLayer(const Coord* const query, Node& resEp, const Idx lc);
		void selectNeighborsHeuristic(FarHeap& outC, const Idx M);

	public:
		Hnsw(const size_t dim, const Idx efConstruction, const Idx M, const Idx maxNodeCount, const unsigned int seed);
		Hnsw(const Hnsw&) = delete;
		Hnsw(Hnsw&&) = delete;
		void insert(const Coord* const coords, const Idx idx);
		void knnSearch(const Coord* const query, const Idx K, const Idx ef, IdxVec& resIndices, CoordVec& resDistances);
		Hnsw& operator=(const Hnsw&) = delete;
		Hnsw& operator=(Hnsw&&) = delete;
	};

	template<typename Coord>
	Coord euclideanDistance(const Coord* const node, const Coord* const query, const size_t dim);

	template<typename Coord>
	Coord innerProductDistance(const Coord* const node, const Coord* const query, const size_t dim);

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::fillHeap(const Coord* const query, const Idx newIdx, IdxVec& eConn, FarHeap& eNewConn) {
		eNewConn.clear();
		eNewConn.reserve(eConn.size());
		eNewConn.push(this->getDistance(this->getCoords(newIdx), query), newIdx);

		for(const auto& idx : eConn)
			eNewConn.push(this->getDistance(this->getCoords(idx), query), idx);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline const Coord* const Hnsw<Coord, Idx, useEuclid>::getCoords(const Idx idx) const {
		return &this->coords[idx * this->dim];
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline Coord Hnsw<Coord, Idx, useEuclid>::getDistance(const Coord* const node, const Coord* const query, const bool useCache, const Idx nodeIdx) {
		if(useCache) {
			auto iter = this->distancesCache.find(nodeIdx);

			if(iter != this->distancesCache.end())
				return iter->second;
		}

		Coord res{};

		if constexpr(useEuclid)
			res = euclideanDistance(node, query, this->dim);
		else
			res = innerProductDistance(node, query, this->dim);

		if(useCache)
			this->distancesCache[nodeIdx] = res;

		return res;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline typename Hnsw<Coord, Idx, useEuclid>::IdxVec& Hnsw<Coord, Idx, useEuclid>::getNeighbors(const Idx idx, const Idx lc) {
		return this->connections[idx][lc];
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline Idx Hnsw<Coord, Idx, useEuclid>::getNewLevel() {
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		return Idx(
			-std::log(dist(this->gen)) * this->mL
		);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::initConnections(const Idx idx, const Idx level) {
		this->connections[idx].resize(level + 1);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::searchLowerLayer(const Coord* const query, Node& ep, const Idx ef, const Idx lc, FarHeap& W) {
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
	inline void Hnsw<Coord, Idx, useEuclid>::searchUpperLayer(const Coord* const query, Node& resEp, const Idx lc) {
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
	inline void Hnsw<Coord, Idx, useEuclid>::selectNeighborsHeuristic(FarHeap& outC, const Idx M) {
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
	inline Hnsw<Coord, Idx, useEuclid>::Hnsw(const size_t dim, const Idx efConstruction, const Idx M, const Idx maxNodeCount, const unsigned int seed)
		: dim(dim), efConstruction(efConstruction), entryIdx(0), entryLevel(0), M(M), mL(1.0 / std::log(1.0 * this->M)), Mmax0(this->M * 2) {

		this->coords.resize(this->dim * maxNodeCount);
		this->gen.seed(seed);
		this->connections.resize(maxNodeCount);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::insert(const Coord* const coords, const Idx idx) {
		std::copy(coords, coords + this->dim, this->coords.begin() + this->nodeCount * this->dim);

		if(idx == 0) {
			this->entryLevel = this->getNewLevel();
			this->nodeCount = 1;
			this->initConnections(this->entryIdx, this->entryLevel);
			return;
		}

		this->distancesCache.clear();
		this->nodeCount++;

		Node ep(this->getDistance(this->getCoords(this->entryIdx), coords, true, this->entryIdx), this->entryIdx);
		const auto L = this->entryLevel;
		const auto l = this->getNewLevel();

		this->initConnections(idx, l);

		for(auto lc = L; lc > l; lc--)
			this->searchUpperLayer(coords, ep, lc);

		for(auto lc = std::min(L, l);; lc--) {
			FarHeap candidates{};
			this->searchLowerLayer(coords, ep, this->efConstruction, lc, candidates);
			this->selectNeighborsHeuristic(candidates, this->M);

			auto& neighbors = this->getNeighbors(idx, lc);
			candidates.extractTo(neighbors);

			// ep = nearest from candidates
			ep = Node(candidates.top());
			const auto layerMmax = !lc ? this->Mmax0 : this->M;

			for(const auto& eIdx : neighbors) {
				auto& eConn = this->getNeighbors(eIdx, lc);

				if(eConn.size() < layerMmax)
					eConn.push_back(idx);
				else {
					this->fillHeap(this->getCoords(eIdx), idx, eConn, candidates);
					this->selectNeighborsHeuristic(candidates, layerMmax);
					candidates.extractTo(eConn);
				}
			}

			if(!lc)
				break;
		}

		if(l > L) {
			this->entryIdx = idx;
			this->entryLevel = l;
		}
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void Hnsw<Coord, Idx, useEuclid>::knnSearch(const Coord* const query, const Idx K, const Idx ef, IdxVec& resIndices, CoordVec& resDistances) {
		this->distancesCache.clear();

		Node ep(this->getDistance(this->getCoords(this->entryIdx), query, true, this->entryIdx), this->entryIdx);
		const auto L = this->entryLevel;

		for(size_t lc = L; lc > 0; lc--)
			this->searchUpperLayer(query, ep, lc);

		FarHeap W{};
		this->searchLowerLayer(query, ep, ef, 0, W);

		while(W.len() > K)
			W.pop();

		const auto len = W.len();
		resDistances.resize(len);
		resIndices.resize(len);

		for(size_t i = len - 1;; i--) {
			{
				const auto& n = W.top();
				resDistances[i] = n.dist;
				resIndices[i] = n.idx;
			}
			W.pop();

			if(!i)
				break;
		}
	}

	template<typename Coord>
	Coord euclideanDistance(const Coord* const node, const Coord* const query, const size_t dim) {
		Coord res = 0;

		for(size_t i = 0; i < dim; i++) {
			auto diff = node[i] - query[i];
			res += diff * diff;
		}

		return res;
	}

	template<typename Coord>
	Coord innerProductDistance(const Coord* const node, const Coord* const query, const size_t dim) {
		Coord res = 0;

		for(size_t i = 0; i < dim; i++)
			res += node[i] * query[i];

		return Coord{1} - res;
	}
}
