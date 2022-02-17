#pragma once
#include <cmath>
#include <random>
#include <utility>
#include "distances.hpp"
#include "Heap.hpp"

namespace chm {
	template<typename Dist>
	class Neighbors {
		Iter<Idx> activeCount;
		Iter<Idx> activeStart;
		std::vector<Idx> layer0;
		const size_t maxLen;
		const size_t maxLen0;
		size_t nodeCount;
		std::vector<std::vector<Idx>> upperLayers;

	public:
		Iter<Idx> begin();
		Iter<Idx> end();
		void fillFrom(FarHeap<Dist>& h);
		void init(const Idx idx, const size_t level);
		size_t len() const;
		Neighbors(const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0);
		void push(const Idx i);
		void use(const Idx idx, const size_t lc);
	};

	template<typename Dist>
	class HnswOptim {
		std::vector<Dist> coords;
		const size_t dim;
		const DistFunc<Dist> distFunc;
		const size_t efConstruction;
		Idx entryIdx;
		size_t entryLevel;
		Node<Dist> ep;
		FarHeap<Dist> farHeap;
		std::default_random_engine gen;
		const size_t M;
		const double mL;
		const size_t Mmax0;
		NearHeap<Dist> nearHeap;
		Neighbors<Dist> neighbors;
		Idx nodeCount;
		std::vector<bool> visited;

		void fillHeap(const Dist* const query, const Dist* const insertedQuery);
		const Dist* const getCoords(const Idx idx) const;
		Dist getDistance(const Dist* const node, const Dist* const query);
		size_t getNewLevel();
		bool insertToVisited(const Idx i);
		void prepareVisited();
		void reserveHeaps(const size_t ef, FarHeap<Dist>& W);
		void resetEp(const Dist* const query);
		void searchLowerLayer(const Dist* const query, const size_t ef, const size_t lc, const bool searching, FarHeap<Dist>& W);
		void searchUpperLayer(const Dist* const query, const size_t lc);
		void selectNeighborsHeuristic(const size_t M);

	public:
		HnswOptim(
			const size_t dim, const DistFunc<Dist> distFunc, const size_t efConstruction,
			const size_t M, const size_t maxNodeCount, const Idx seed
		);
		void insert(const Dist* const query);
		FarHeap<Dist> search(const Dist* const query, const size_t K, const size_t ef);
	};

	template<typename Dist>
	inline Iter<Idx> Neighbors<Dist>::begin() {
		return this->activeStart;
	}

	template<typename Dist>
	inline Iter<Idx> Neighbors<Dist>::end() {
		return this->activeStart + *this->activeCount;
	}

	template<typename Dist>
	inline void Neighbors<Dist>::fillFrom(FarHeap<Dist>& h) {
		const auto lastIdx = h.len() - 1;
		*this->activeCount = Idx(h.len());

		for(size_t i = 0; i < lastIdx; i++) {
			*(this->activeStart + i) = h.top().idx;
			h.pop();
		}

		*(this->activeStart + lastIdx) = h.top().idx;
	}

	template<typename Dist>
	inline void Neighbors<Dist>::init(const Idx idx, const size_t level) {
		this->nodeCount++;

		if(level)
			this->upperLayers[idx].resize(this->maxLen * level, 0);
	}

	template<typename Dist>
	inline size_t Neighbors<Dist>::len() const {
		return *this->activeCount;
	}

	template<typename Dist>
	inline Neighbors<Dist>::Neighbors(const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0)
		: maxLen(Mmax + 1), maxLen0(Mmax0 + 1), nodeCount(0) {

		this->layer0.resize(maxNodeCount * this->maxLen0, 0);
		this->upperLayers.resize(maxNodeCount);
	}

	template<typename Dist>
	inline void Neighbors<Dist>::push(const Idx i) {
		*(this->activeStart + *this->activeCount) = i;
		(*this->activeCount)++;
	}

	template<typename Dist>
	inline void Neighbors<Dist>::use(const Idx idx, const size_t lc) {
		this->activeCount = lc ? this->upperLayers[idx].begin() + this->maxLen * (lc - 1) : this->layer0.begin() + this->maxLen0 * idx;
		this->activeStart = this->activeCount + 1;
	}

	template<typename Dist>
	inline void HnswOptim<Dist>::fillHeap(const Dist* const query, const Dist* const insertedQuery) {
		this->farHeap.clear();
		this->farHeap.push(this->getDistance(insertedQuery, query), this->nodeCount);

		for(const auto& idx : this->neighbors)
			this->farHeap.push(this->getDistance(this->getCoords(idx), query), idx);
	}

	template<typename Dist>
	inline const Dist* const HnswOptim<Dist>::getCoords(const Idx idx) const {
		return this->coords.data() + idx * this->dim;
	}

	template<typename Dist>
	inline Dist HnswOptim<Dist>::getDistance(const Dist* const node, const Dist* const query) {
		return this->distFunc(node, query, this->dim);
	}

	template<typename Dist>
	inline size_t HnswOptim<Dist>::getNewLevel() {
		std::uniform_real_distribution<double> dist(0.0, 1.0);
		return size_t(-std::log(dist(this->gen)) * this->mL);
	}

	template<typename Dist>
	inline bool HnswOptim<Dist>::insertToVisited(const Idx i) {
		if(this->visited[i])
			return false;

		this->visited[i] = true;
		return true;
	}

	template<typename Dist>
	inline void HnswOptim<Dist>::prepareVisited() {
		this->visited.clear();
		this->visited.resize(this->nodeCount);
		this->visited[this->ep.idx] = true;
	}

	template<typename Dist>
	inline void HnswOptim<Dist>::reserveHeaps(const size_t ef, FarHeap<Dist>& W) {
		const auto maxLen = std::max(ef, this->Mmax0);
		W.reserve(maxLen);
		this->nearHeap.reserve(maxLen);
	}

	template<typename Dist>
	inline void HnswOptim<Dist>::resetEp(const Dist* const query) {
		this->ep.dist = this->getDistance(this->getCoords(this->entryIdx), query);
		this->ep.idx = this->entryIdx;
	}

	template<typename Dist>
	inline void HnswOptim<Dist>::searchLowerLayer(const Dist* const query, const size_t ef, const size_t lc, const bool searching, FarHeap<Dist>& W) {
		auto& C = this->nearHeap;

		C.clear();
		C.push(this->ep);
		this->prepareVisited();
		W.clear();
		W.push(this->ep);

		while(C.len()) {
			Idx cIdx{};

			{
				const auto& c = C.top();
				const auto& f = W.top();

				if(c.dist > f.dist && (searching || W.len() == ef))
					break;

				cIdx = c.idx;
			}

			this->neighbors.use(cIdx, lc);

			// Extract nearest from C.
			C.pop();

			for(const auto& eIdx : this->neighbors) {
				if(this->insertToVisited(eIdx)) {
					const auto eDist = this->getDistance(this->getCoords(eIdx), query);
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
	inline void HnswOptim<Dist>::searchUpperLayer(const Dist* const query, const size_t lc) {
		Idx prevIdx{};

		do {
			this->neighbors.use(this->ep.idx, lc);
			prevIdx = this->ep.idx;

			for(const auto& cIdx : this->neighbors) {
				const auto cDist = this->getDistance(this->getCoords(cIdx), query);

				if(cDist < this->ep.dist) {
					this->ep.dist = cDist;
					this->ep.idx = cIdx;
				}
			}

		} while(this->ep.idx != prevIdx);
	}

	template<typename Dist>
	inline void HnswOptim<Dist>::selectNeighborsHeuristic(const size_t M) {
		if(this->farHeap.len() < M)
			return;

		auto& R = this->farHeap;
		auto& W = this->nearHeap;

		W.loadFrom(R);

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
	inline HnswOptim<Dist>::HnswOptim(
		const size_t dim, const DistFunc<Dist> distFunc, const size_t efConstruction,
		const size_t M, const size_t maxNodeCount, const Idx seed
	) : dim(dim), distFunc(distFunc), efConstruction(efConstruction), entryIdx(0), entryLevel(0),
		M(M), mL(1.0 / std::log(1.0 * this->M)), Mmax0(this->M * 2), neighbors(maxNodeCount, this->M, this->Mmax0), nodeCount(0) {

		this->coords.resize(this->dim * maxNodeCount);
		this->gen.seed(seed);
		this->reserveHeaps(this->efConstruction, this->farHeap);
	}

	template<typename Dist>
	inline void HnswOptim<Dist>::insert(const Dist* const query) {
		std::copy(query, query + this->dim, this->coords.begin() + this->nodeCount * this->dim);

		if(!this->nodeCount) {
			this->entryLevel = this->getNewLevel();
			this->nodeCount = 1;
			this->neighbors.init(this->entryIdx, this->entryLevel);
			return;
		}

		this->resetEp(query);
		const auto L = this->entryLevel;
		const auto l = this->getNewLevel();

		this->neighbors.init(this->nodeCount, l);

		for(auto lc = L; lc > l; lc--)
			this->searchUpperLayer(query, lc);

		for(auto lc = std::min(L, l);; lc--) {
			this->searchLowerLayer(query, this->efConstruction, lc, false, this->farHeap);
			this->selectNeighborsHeuristic(this->M);

			this->neighbors.use(this->nodeCount, lc);
			this->neighbors.fillFrom(this->farHeap);

			// ep = nearest from candidates
			{
				const auto& n = this->farHeap.top();
				this->ep.dist = n.dist;
				this->ep.idx = n.idx;
			}
			const auto layerMmax = !lc ? this->Mmax0 : this->M;

			for(const auto& eIdx : this->neighbors) {
				this->neighbors.use(eIdx, lc);

				if(this->neighbors.len() < layerMmax)
					this->neighbors.push(this->nodeCount);
				else {
					this->fillHeap(this->getCoords(eIdx), query);
					this->selectNeighborsHeuristic(layerMmax);
					this->neighbors.fillFrom(this->farHeap);
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
	inline FarHeap<Dist> HnswOptim<Dist>::search(const Dist* const query, const size_t K, const size_t ef) {
		FarHeap<Dist> W{};
		this->reserveHeaps(ef, W);

		this->resetEp(query);
		const auto L = this->entryLevel;

		for(size_t lc = L; lc > 0; lc--)
			this->searchUpperLayer(query, lc);

		this->searchLowerLayer(query, ef, 0, true, W);

		while(W.len() > K)
			W.pop();

		return W;
	}
}
