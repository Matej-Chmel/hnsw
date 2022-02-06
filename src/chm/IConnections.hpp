#pragma once
#include "Unique.hpp"

namespace chm {
	using IdxVec3D = std::vector<std::vector<std::vector<size_t>>>;
	using IdxVec3DPtr = std::shared_ptr<IdxVec3D>;

	template<typename Idx>
	struct INeighbors : public Unique {
		virtual ~INeighbors() = default;
		virtual ConstIter<Idx> begin() const = 0;
		virtual void clear() = 0;
		virtual ConstIter<Idx> end() const = 0;
		virtual size_t len() const = 0;
		virtual void push(const Idx i) = 0;
		virtual void reserve(const size_t capacity) = 0;
	};

	template<typename Idx>
	using INeighborsPtr = std::shared_ptr<INeighbors<Idx>>;

	template<typename Idx>
	struct IConnections : public Unique {
		virtual ~IConnections() = default;
		virtual IdxVec3DPtr getConnections() = 0;
		virtual INeighborsPtr<Idx> getNeighbors(const Idx idx, const size_t lc) = 0;
		virtual void init(const Idx idx, const size_t level) = 0;
	};

	template<typename Idx>
	using IConnPtr = std::shared_ptr<IConnections<Idx>>;

	template<typename Idx>
	class Neighbors3D : public INeighbors<Idx> {
		std::vector<Idx>* const v;

	public:
		ConstIter<Idx> begin() const override;
		void clear() override;
		ConstIter<Idx> end() const override;
		size_t len() const override;
		Neighbors3D(std::vector<Idx>* const v);
		void push(const Idx i) override;
		void reserve(const size_t capacity) override;
	};

	template<typename Idx>
	class Connections3D : public IConnections<Idx> {
		std::vector<std::vector<std::vector<Idx>>> c;
		size_t nodeCount;

	public:
		Connections3D(const size_t maxNodeCount);
		IdxVec3DPtr getConnections() override;
		INeighborsPtr<Idx> getNeighbors(const Idx idx, const size_t lc) override;
		void init(const Idx idx, const size_t level) override;
	};

	template<typename Idx>
	class NeighborsLayer0 : public INeighbors<Idx> {
		Iter<Idx> count;
		Iter<Idx> start;

	public:
		ConstIter<Idx> begin() const override;
		void clear() override;
		ConstIter<Idx> end() const override;
		size_t len() const override;
		NeighborsLayer0(const Iter<Idx>& count, const Iter<Idx>& start);
		void push(const Idx i) override;
		void reserve(const size_t capacity) override;
	};

	template<typename Idx>
	class ConnLayer0 : public IConnections<Idx> {
		std::vector<Idx> layer0;
		std::vector<Idx> levels;
		const size_t maxLen;
		const size_t maxLen0;
		size_t nodeCount;
		std::vector<std::vector<Idx>> upperLayers;

	public:
		ConnLayer0(const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0);
		IdxVec3DPtr getConnections() override;
		INeighborsPtr<Idx> getNeighbors(const Idx idx, const size_t lc) override;
		void init(const Idx idx, const size_t level) override;
	};

	template<typename Idx>
	IConnPtr<Idx> createConn(const bool useConnLayer0, const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0);

	template<typename Idx>
	inline ConstIter<Idx> Neighbors3D<Idx>::begin() const {
		return this->v->begin();
	}

	template<typename Idx>
	inline void Neighbors3D<Idx>::clear() {
		this->v->clear();
	}

	template<typename Idx>
	inline ConstIter<Idx> Neighbors3D<Idx>::end() const {
		return this->v->end();
	}

	template<typename Idx>
	inline size_t Neighbors3D<Idx>::len() const {
		return this->v->size();
	}

	template<typename Idx>
	inline Neighbors3D<Idx>::Neighbors3D(std::vector<Idx>* const v) : v(v) {}

	template<typename Idx>
	inline void Neighbors3D<Idx>::push(const Idx i) {
		this->v->emplace_back(i);
	}

	template<typename Idx>
	inline void Neighbors3D<Idx>::reserve(const size_t capacity) {
		this->v->reserve(capacity);
	}

	template<typename Idx>
	inline Connections3D<Idx>::Connections3D(const size_t maxNodeCount) : nodeCount(0) {
		this->c.resize(maxNodeCount);
	}

	template<typename Idx>
	inline IdxVec3DPtr Connections3D<Idx>::getConnections() {
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

	template<typename Idx>
	inline INeighborsPtr<Idx> Connections3D<Idx>::getNeighbors(const Idx idx, const size_t lc) {
		return std::make_shared<Neighbors3D<Idx>>(&this->c[idx][lc]);
	}

	template<typename Idx>
	inline void Connections3D<Idx>::init(const Idx idx, const size_t level) {
		this->c[idx].resize(level + 1);
		this->nodeCount++;
	}

	template<typename Idx>
	inline ConstIter<Idx> NeighborsLayer0<Idx>::begin() const {
		return this->start;
	}

	template<typename Idx>
	inline void NeighborsLayer0<Idx>::clear() {
		*this->count = 0;
	}

	template<typename Idx>
	inline ConstIter<Idx> NeighborsLayer0<Idx>::end() const {
		return this->start + *this->count;
	}

	template<typename Idx>
	inline size_t NeighborsLayer0<Idx>::len() const {
		return size_t(*this->count);
	}

	template<typename Idx>
	inline NeighborsLayer0<Idx>::NeighborsLayer0(const Iter<Idx>& count, const Iter<Idx>& start) : count(count), start(start) {}

	template<typename Idx>
	inline void NeighborsLayer0<Idx>::push(const Idx i) {
		*(this->start + *this->count) = i;
		(*this->count)++;
	}

	template<typename Idx>
	inline void NeighborsLayer0<Idx>::reserve(const size_t) {}

	template<typename Idx>
	inline ConnLayer0<Idx>::ConnLayer0(const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0)
		: maxLen(Mmax + 1), maxLen0(Mmax0 + 1), nodeCount(0) {
		this->layer0.resize(maxNodeCount * (this->maxLen0), 0);
		this->levels.resize(maxNodeCount);
		this->upperLayers.resize(maxNodeCount);
	}

	template<typename Idx>
	inline IdxVec3DPtr ConnLayer0<Idx>::getConnections() {
		auto res = std::make_shared<IdxVec3D>();
		auto& r = *res;
		r.resize(this->nodeCount);

		for(size_t i = 0; i < this->nodeCount; i++) {
			auto& nodeLayers = r[i];
			const auto nodeLayersLen = size_t(this->levels[i]) + 1;
			nodeLayers.resize(nodeLayersLen);

			for(size_t level = 0; level < nodeLayersLen; level++) {
				const auto neighbors = this->getNeighbors(i, level);
				auto& resNeighbors = nodeLayers[level];
				resNeighbors.reserve(neighbors->len());

				for(const auto& n : *neighbors)
					resNeighbors.emplace_back(n);
			}
		}

		return res;
	}

	template<typename Idx>
	inline INeighborsPtr<Idx> ConnLayer0<Idx>::getNeighbors(const Idx idx, const size_t lc) {
		Iter<Idx> count;

		if(lc)
			count = this->upperLayers[idx].begin() + this->maxLen * (lc - 1);
		else
			count = this->layer0.begin() + this->maxLen0 * idx;

		return std::make_shared<NeighborsLayer0<Idx>>(count, count + 1);
	}

	template<typename Idx>
	inline void ConnLayer0<Idx>::init(const Idx idx, const size_t level) {
		this->levels[idx] = level;
		this->nodeCount++;

		if(level)
			this->upperLayers[idx].resize(this->maxLen * level, 0);
	}

	template<typename Idx>
	IConnPtr<Idx> createConn(const bool useConnLayer0, const size_t maxNodeCount, const size_t Mmax, const size_t Mmax0) {
		if(useConnLayer0)
			return std::make_shared<ConnLayer0<Idx>>(maxNodeCount, Mmax, Mmax0);
		return std::make_shared<Connections3D<Idx>>(maxNodeCount);
	}
}
