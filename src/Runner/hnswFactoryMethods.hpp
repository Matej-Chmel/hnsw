#pragma once
#include "HnswIntermediate.hpp"

namespace chm {
	enum class HnswKind {
		CHM_AUTO,
		CHM_INT,
		CHM_SIZE_T,
		CHM_SHORT,
		HNSWLIB
	};

	struct HnswType : public Unique {
		const bool isIntermediate;
		const HnswKind kind;

		HnswType(const bool isIntermediate, const HnswKind kind);
	};

	using HnswTypePtr = std::shared_ptr<HnswType>;

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg);

	template<typename Coord, bool useEuclid>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg);

	template<typename Coord, typename Idx>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg);

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg, const HnswTypePtr& type);

	template<typename Coord, typename Idx>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg);

	template<typename Coord>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg, const HnswTypePtr& type);

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg) {
		if(cfg->useEuclid)
			return createHnswIntermediate<Coord, true>(cfg);
		return createHnswIntermediate<Coord, false>(cfg);
	}

	template<typename Coord, bool useEuclid>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg) {
		if(isWideEnough<unsigned short>(cfg->maxNodeCount))
			return std::make_shared<HnswInterImpl<Coord, unsigned short, useEuclid>>(cfg);
		if(isWideEnough<unsigned int>(cfg->maxNodeCount))
			return std::make_shared<HnswInterImpl<Coord, unsigned int, useEuclid>>(cfg);
		return std::make_shared<HnswInterImpl<Coord, size_t, useEuclid>>(cfg);
	}

	template<typename Coord, typename Idx>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg) {
		if(cfg->useEuclid)
			return std::make_shared<HnswInterImpl<Coord, Idx, true>>(cfg);
		return std::make_shared<HnswInterImpl<Coord, Idx, false>>(cfg);
	}

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg, const HnswTypePtr& type) {
		switch(type->kind) {
			case HnswKind::CHM_AUTO:
				return createHnswIntermediate<Coord>(cfg);
			case HnswKind::CHM_INT:
				return createHnswIntermediate<Coord, unsigned int>(cfg);
			case HnswKind::CHM_SHORT:
				return createHnswIntermediate<Coord, unsigned short>(cfg);
			case HnswKind::CHM_SIZE_T:
				return createHnswIntermediate<Coord, size_t>(cfg);
			case HnswKind::HNSWLIB:
				return std::make_shared<HnswlibInterImpl<Coord>>(cfg);
			default:
				throw std::runtime_error("Unknown HnswKind value.");
		}
	}

	template<typename Coord, typename Idx>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg) {
		if(cfg->useEuclid)
			return std::make_shared<Hnsw<Coord, Idx, true>>(cfg);
		return std::make_shared<Hnsw<Coord, Idx, false>>(cfg);
	}

	template<typename Coord>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg, const HnswTypePtr& type) {
		if(type->isIntermediate)
			return createHnswIntermediate<Coord>(cfg, type);

		switch(type->kind) {
			case HnswKind::CHM_AUTO:
				return createHnsw<Coord>(cfg);
			case HnswKind::CHM_INT:
				return createHnsw<Coord, unsigned int>(cfg);
			case HnswKind::CHM_SHORT:
				return createHnsw<Coord, unsigned short>(cfg);
			case HnswKind::CHM_SIZE_T:
				return createHnsw<Coord, size_t>(cfg);
			case HnswKind::HNSWLIB:
				return std::make_shared<HnswlibWrapper<Coord>>(cfg);
			default:
				throw std::runtime_error("Unknown HnswKind value.");
		}
	}
}
