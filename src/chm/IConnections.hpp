#pragma once
#include "Heap.hpp"

namespace chm {
	using IdxVec3D = std::vector<std::vector<std::vector<size_t>>>;
	using IdxVec3DPtr = std::shared_ptr<IdxVec3D>;

	template<typename Coord, typename Idx>
	struct IConnections : public Unique {
		virtual ~IConnections() = default;
		virtual Iter<Idx> begin() = 0;
		virtual Iter<Idx> end() = 0;
		virtual void fillFrom(FarHeap<Coord, Idx>& h) = 0;
		virtual IdxVec3DPtr getConnections() = 0;
		virtual void init(const Idx idx, const size_t level) = 0;
		virtual size_t len() const = 0;
		virtual void push(const Idx i) = 0;
		virtual void useNeighbors(const Idx idx, const size_t lc) = 0;
	};

	template<typename Coord, typename Idx>
	using IConnPtr = std::shared_ptr<IConnections<Coord, Idx>>;

	template<typename Coord, typename Idx>
	class Connections3D : public IConnections<Coord, Idx> {
		std::vector<std::vector<std::vector<Idx>>> c;
		std::vector<Idx>* neighbors;
		size_t nodeCount;

	public:
		Iter<Idx> begin() override;
		Connections3D(const size_t maxNodeCount);
		Iter<Idx> end() override;
		void fillFrom(FarHeap<Coord, Idx>& h) override;
		IdxVec3DPtr getConnections() override;
		void init(const Idx idx, const size_t level) override;
		size_t len() const override;
		void push(const Idx i) override;
		void useNeighbors(const Idx idx, const size_t lc) override;
	};

	template<typename Coord, typename Idx>
	class ConnLayer0 : public IConnections<Coord, Idx> {
		std::vector<Idx> layer0;
		std::vector<Idx> levels;
		const size_t maxLen;
		const size_t maxLen0;
		Iter<Idx> neighborCount;
		Iter<Idx> neighborStart;
		size_t nodeCount;
		std::vector<std::vector<Idx>> upperLayers;

	public:
		Iter<Idx> begin() override;
		ConnLayer0(const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0);
		Iter<Idx> end() override;
		void fillFrom(FarHeap<Coord, Idx>& h) override;
		IdxVec3DPtr getConnections() override;
		void init(const Idx idx, const size_t level) override;
		size_t len() const override;
		void push(const Idx i) override;
		void useNeighbors(const Idx idx, const size_t lc) override;
	};

	template<typename Coord, typename Idx>
	IConnPtr<Coord, Idx> createConn(const bool useConnLayer0, const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0);

	template<typename Coord, typename Idx>
	inline Iter<Idx> Connections3D<Coord, Idx>::begin() {
		return this->neighbors->begin();
	}

	template<typename Coord, typename Idx>
	inline Connections3D<Coord, Idx>::Connections3D(const size_t maxNodeCount) : nodeCount(0) {
		this->c.resize(maxNodeCount);
	}

	template<typename Coord, typename Idx>
	inline Iter<Idx> Connections3D<Coord, Idx>::end() {
		return this->neighbors->end();
	}

	template<typename Coord, typename Idx>
	inline void Connections3D<Coord, Idx>::fillFrom(FarHeap<Coord, Idx>& h) {
		this->neighbors->clear();
		this->neighbors->reserve(h.len());

		while(h.len() > 1) {
			this->neighbors->emplace_back(h.top().idx);
			h.pop();
		}

		this->neighbors->emplace_back(h.top().idx);
	}

	template<typename Coord, typename Idx>
	inline IdxVec3DPtr Connections3D<Coord, Idx>::getConnections() {
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
	inline void Connections3D<Coord, Idx>::init(const Idx idx, const size_t level) {
		this->c[idx].resize(level + 1);
		this->nodeCount++;
	}

	template<typename Coord, typename Idx>
	inline size_t Connections3D<Coord, Idx>::len() const {
		return this->neighbors->size();
	}

	template<typename Coord, typename Idx>
	inline void Connections3D<Coord, Idx>::push(const Idx i) {
		this->neighbors->emplace_back(i);
	}

	template<typename Coord, typename Idx>
	inline void Connections3D<Coord, Idx>::useNeighbors(const Idx idx, const size_t lc) {
		this->neighbors = &this->c[idx][lc];
	}

	template<typename Coord, typename Idx>
	inline Iter<Idx> ConnLayer0<Coord, Idx>::begin() {
		return this->neighborStart;
	}

	template<typename Coord, typename Idx>
	inline ConnLayer0<Coord, Idx>::ConnLayer0(const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0)
		: maxLen(Mmax + 1), maxLen0(Mmax0 + 1), nodeCount(0) {
		this->layer0.resize(maxNodeCount * (this->maxLen0), 0);
		this->levels.resize(maxNodeCount);
		this->upperLayers.resize(maxNodeCount);
	}

	template<typename Coord, typename Idx>
	inline Iter<Idx> ConnLayer0<Coord, Idx>::end() {
		return this->neighborStart + *this->neighborCount;
	}

	template<typename Coord, typename Idx>
	inline void ConnLayer0<Coord, Idx>::fillFrom(FarHeap<Coord, Idx>& h) {
		const auto lastIdx = h.len() - 1;
		*this->neighborCount = Idx(h.len());

		for(size_t i = 0; i < lastIdx; i++) {
			*(this->neighborStart + i) = h.top().idx;
			h.pop();
		}

		*(this->neighborStart + lastIdx) = h.top().idx;
	}

	template<typename Coord, typename Idx>
	inline IdxVec3DPtr ConnLayer0<Coord, Idx>::getConnections() {
		auto res = std::make_shared<IdxVec3D>();
		auto& r = *res;
		r.resize(this->nodeCount);

		for(size_t i = 0; i < this->nodeCount; i++) {
			auto& nodeLayers = r[i];
			const auto nodeLayersLen = size_t(this->levels[i]) + 1;
			nodeLayers.resize(nodeLayersLen);

			for(size_t level = 0; level < nodeLayersLen; level++) {
				this->useNeighbors(i, level);

				auto& neighbors = nodeLayers[level];
				neighbors.reserve(this->len());

				for(const auto& n : *this)
					neighbors.emplace_back(n);
			}
		}

		return res;
	}

	template<typename Coord, typename Idx>
	inline void ConnLayer0<Coord, Idx>::init(const Idx idx, const size_t level) {
		this->levels[idx] = level;
		this->nodeCount++;

		if(level)
			this->upperLayers[idx].resize(this->maxLen * level, 0);
	}

	template<typename Coord, typename Idx>
	inline size_t ConnLayer0<Coord, Idx>::len() const {
		return *this->neighborCount;
	}

	template<typename Coord, typename Idx>
	inline void ConnLayer0<Coord, Idx>::push(const Idx i) {
		*(this->neighborStart + *this->neighborCount) = i;
		(*this->neighborCount)++;
	}

	template<typename Coord, typename Idx>
	inline void ConnLayer0<Coord, Idx>::useNeighbors(const Idx idx, const size_t lc) {
		this->neighborCount = lc ? this->upperLayers[idx].begin() + this->maxLen * (lc - 1) : this->layer0.begin() + this->maxLen0 * idx;
		this->neighborStart = this->neighborCount + 1;
	}

	template<typename Coord, typename Idx>
	IConnPtr<Coord, Idx> createConn(const bool useConnLayer0, const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0) {
		if(useConnLayer0)
			return std::make_shared<ConnLayer0<Coord, Idx>>(maxNodeCount, Mmax, Mmax0);
		return std::make_shared<Connections3D<Coord, Idx>>(maxNodeCount);
	}
}
