#pragma once
#include "Heap.hpp"

namespace chm {
	using IdxVec3D = std::vector<std::vector<std::vector<size_t>>>;
	using IdxVec3DPtr = std::shared_ptr<IdxVec3D>;

	template<typename Coord, typename Idx>
	struct INeighbors : public Unique {
		virtual ~INeighbors() = default;
		virtual Iter<Idx> begin() = 0;
		virtual Iter<Idx> end() = 0;
		virtual void fillFrom(FarHeap<Coord, Idx>& h) = 0;
		virtual IdxVec3DPtr getConnections() = 0;
		virtual void init(const Idx idx, const size_t level) = 0;
		virtual size_t len() const = 0;
		virtual void push(const Idx i) = 0;
		virtual void use(const Idx idx, const size_t lc) = 0;
	};

	template<typename Coord, typename Idx>
	using INeighborsPtr = std::shared_ptr<INeighbors<Coord, Idx>>;

	template<typename Coord, typename Idx>
	class Neighbors3D : public INeighbors<Coord, Idx> {
		std::vector<Idx>* active;
		std::vector<std::vector<std::vector<Idx>>> c;
		size_t nodeCount;

	public:
		Iter<Idx> begin() override;
		Iter<Idx> end() override;
		void fillFrom(FarHeap<Coord, Idx>& h) override;
		IdxVec3DPtr getConnections() override;
		void init(const Idx idx, const size_t level) override;
		size_t len() const override;
		Neighbors3D(const size_t maxNodeCount);
		void push(const Idx i) override;
		void use(const Idx idx, const size_t lc) override;
	};

	template<typename Coord, typename Idx>
	class PreAllocNeighbors : public INeighbors<Coord, Idx> {
		Iter<Idx> activeCount;
		Iter<Idx> activeStart;
		std::vector<Idx> layer0;
		std::vector<Idx> levels;
		const size_t maxLen;
		const size_t maxLen0;
		size_t nodeCount;
		std::vector<std::vector<Idx>> upperLayers;

	public:
		Iter<Idx> begin() override;
		Iter<Idx> end() override;
		void fillFrom(FarHeap<Coord, Idx>& h) override;
		IdxVec3DPtr getConnections() override;
		void init(const Idx idx, const size_t level) override;
		size_t len() const override;
		PreAllocNeighbors(const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0);
		void push(const Idx i) override;
		void use(const Idx idx, const size_t lc) override;
	};

	template<typename Coord, typename Idx>
	INeighborsPtr<Coord, Idx> createNeighbors(const bool usePreAlloc, const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0);

	template<typename Coord, typename Idx>
	inline Iter<Idx> Neighbors3D<Coord, Idx>::begin() {
		return this->active->begin();
	}

	template<typename Coord, typename Idx>
	inline Iter<Idx> Neighbors3D<Coord, Idx>::end() {
		return this->active->end();
	}

	template<typename Coord, typename Idx>
	inline void Neighbors3D<Coord, Idx>::fillFrom(FarHeap<Coord, Idx>& h) {
		this->active->clear();
		this->active->reserve(h.len());

		while(h.len() > 1) {
			this->active->emplace_back(h.top().idx);
			h.pop();
		}

		this->active->emplace_back(h.top().idx);
	}

	template<typename Coord, typename Idx>
	inline IdxVec3DPtr Neighbors3D<Coord, Idx>::getConnections() {
		auto res = std::make_shared<IdxVec3D>();
		auto& r = *res;
		r.resize(this->nodeCount);

		for(size_t i = 0; i < this->nodeCount; i++) {
			const auto& nodeLayers = this->c[i];
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

	template<typename Coord, typename Idx>
	inline void Neighbors3D<Coord, Idx>::init(const Idx idx, const size_t level) {
		this->c[idx].resize(level + 1);
		this->nodeCount++;
	}

	template<typename Coord, typename Idx>
	inline size_t Neighbors3D<Coord, Idx>::len() const {
		return this->active->size();
	}

	template<typename Coord, typename Idx>
	inline Neighbors3D<Coord, Idx>::Neighbors3D(const size_t maxNodeCount) : nodeCount(0) {
		this->c.resize(maxNodeCount);
	}

	template<typename Coord, typename Idx>
	inline void Neighbors3D<Coord, Idx>::push(const Idx i) {
		this->active->emplace_back(i);
	}

	template<typename Coord, typename Idx>
	inline void Neighbors3D<Coord, Idx>::use(const Idx idx, const size_t lc) {
		this->active = &this->c[idx][lc];
	}

	template<typename Coord, typename Idx>
	inline Iter<Idx> PreAllocNeighbors<Coord, Idx>::begin() {
		return this->activeStart;
	}

	template<typename Coord, typename Idx>
	inline Iter<Idx> PreAllocNeighbors<Coord, Idx>::end() {
		return this->activeStart + *this->activeCount;
	}

	template<typename Coord, typename Idx>
	inline void PreAllocNeighbors<Coord, Idx>::fillFrom(FarHeap<Coord, Idx>& h) {
		const auto lastIdx = h.len() - 1;
		*this->activeCount = Idx(h.len());

		for(size_t i = 0; i < lastIdx; i++) {
			*(this->activeStart + i) = h.top().idx;
			h.pop();
		}

		*(this->activeStart + lastIdx) = h.top().idx;
	}

	template<typename Coord, typename Idx>
	inline IdxVec3DPtr PreAllocNeighbors<Coord, Idx>::getConnections() {
		auto res = std::make_shared<IdxVec3D>();
		auto& r = *res;
		r.resize(this->nodeCount);

		for(size_t i = 0; i < this->nodeCount; i++) {
			auto& nodeLayers = r[i];
			const auto nodeLayersLen = size_t(this->levels[i]) + 1;
			nodeLayers.resize(nodeLayersLen);

			for(size_t level = 0; level < nodeLayersLen; level++) {
				this->use(i, level);

				auto& neighbors = nodeLayers[level];
				neighbors.reserve(this->len());

				for(const auto& n : *this)
					neighbors.emplace_back(n);
			}
		}

		return res;
	}

	template<typename Coord, typename Idx>
	inline void PreAllocNeighbors<Coord, Idx>::init(const Idx idx, const size_t level) {
		this->levels[idx] = level;
		this->nodeCount++;

		if(level)
			this->upperLayers[idx].resize(this->maxLen * level, 0);
	}

	template<typename Coord, typename Idx>
	inline size_t PreAllocNeighbors<Coord, Idx>::len() const {
		return *this->activeCount;
	}

	template<typename Coord, typename Idx>
	inline PreAllocNeighbors<Coord, Idx>::PreAllocNeighbors(const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0)
		: maxLen(Mmax + 1), maxLen0(Mmax0 + 1), nodeCount(0) {
		this->layer0.resize(maxNodeCount * (this->maxLen0), 0);
		this->levels.resize(maxNodeCount);
		this->upperLayers.resize(maxNodeCount);
	}

	template<typename Coord, typename Idx>
	inline void PreAllocNeighbors<Coord, Idx>::push(const Idx i) {
		*(this->activeStart + *this->activeCount) = i;
		(*this->activeCount)++;
	}

	template<typename Coord, typename Idx>
	inline void PreAllocNeighbors<Coord, Idx>::use(const Idx idx, const size_t lc) {
		this->activeCount = lc ? this->upperLayers[idx].begin() + this->maxLen * (lc - 1) : this->layer0.begin() + this->maxLen0 * idx;
		this->activeStart = this->activeCount + 1;
	}

	template<typename Coord, typename Idx>
	INeighborsPtr<Coord, Idx> createNeighbors(const bool usePreAlloc, const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0) {
		if(usePreAlloc)
			return std::make_shared<PreAllocNeighbors<Coord, Idx>>(maxNodeCount, Mmax, Mmax0);
		return std::make_shared<Neighbors3D<Coord, Idx>>(maxNodeCount);
	}
}
