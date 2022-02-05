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
		const HnswCfgPtr cfg;
		const bool isIntermediate;
		const HnswKind kind;
		const HnswSettingsPtr settings;

		HnswType(const HnswCfgPtr& cfg, const bool isIntermediate, const HnswKind kind, const HnswSettingsPtr& settings = nullptr);
	};

	using HnswTypePtr = std::shared_ptr<HnswType>;

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createChmHnswInter(const HnswTypePtr& type);

	template<typename Coord, bool useEuclid>
	IHnswIntermediatePtr<Coord> createChmHnswInter(const HnswTypePtr& type);

	template<typename Coord, typename Idx>
	IHnswIntermediatePtr<Coord> createChmHnswInter(const HnswTypePtr& type);

	template<typename Coord, typename Idx>
	IHnswPtr<Coord> createChmHnsw(const HnswTypePtr& type);

	template<typename Coord>
	IHnswPtr<Coord> createHnsw(const HnswTypePtr& type);

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createHnswInter(const HnswTypePtr& type);

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createChmHnswInter(const HnswTypePtr& type) {
		if(type->cfg->useEuclid)
			return createChmHnswInter<Coord, true>(type);
		return createChmHnswInter<Coord, false>(type);
	}

	template<typename Coord, bool useEuclid>
	IHnswIntermediatePtr<Coord> createChmHnswInter(const HnswTypePtr& type) {
		if(isWideEnough<unsigned short>(type->cfg->maxNodeCount))
			return std::make_shared<HnswInterImpl<Coord, unsigned short, useEuclid>>(type->cfg, type->settings);
		if(isWideEnough<unsigned int>(type->cfg->maxNodeCount))
			return std::make_shared<HnswInterImpl<Coord, unsigned int, useEuclid>>(type->cfg, type->settings);
		return std::make_shared<HnswInterImpl<Coord, size_t, useEuclid>>(type->cfg, type->settings);
	}

	template<typename Coord, typename Idx>
	IHnswIntermediatePtr<Coord> createChmHnswInter(const HnswTypePtr& type) {
		if(type->cfg->useEuclid)
			return std::make_shared<HnswInterImpl<Coord, Idx, true>>(type->cfg, type->settings);
		return std::make_shared<HnswInterImpl<Coord, Idx, false>>(type->cfg, type->settings);
	}

	template<typename Coord, typename Idx>
	IHnswPtr<Coord> createChmHnsw(const HnswTypePtr& type) {
		if(type->cfg->useEuclid)
			return std::make_shared<Hnsw<Coord, Idx, true>>(type->cfg, type->settings);
		return std::make_shared<Hnsw<Coord, Idx, false>>(type->cfg, type->settings);
	}

	template<typename Coord>
	IHnswPtr<Coord> createHnsw(const HnswTypePtr& type) {
		if(type->isIntermediate)
			return createHnswInter<Coord>(type);

		switch(type->kind) {
			case HnswKind::CHM_AUTO:
				return createHnsw<Coord>(type->cfg, type->settings);
			case HnswKind::CHM_INT:
				return createChmHnsw<Coord, unsigned int>(type);
			case HnswKind::CHM_SHORT:
				return createChmHnsw<Coord, unsigned short>(type);
			case HnswKind::CHM_SIZE_T:
				return createChmHnsw<Coord, size_t>(type);
			case HnswKind::HNSWLIB:
				return std::make_shared<HnswlibWrapper<Coord>>(type->cfg);
			default:
				throw std::runtime_error("Unknown HnswKind value.");
		}
	}

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createHnswInter(const HnswTypePtr& type) {
		switch(type->kind) {
			case HnswKind::CHM_AUTO:
				return createChmHnswInter<Coord>(type);
			case HnswKind::CHM_INT:
				return createChmHnswInter<Coord, unsigned int>(type);
			case HnswKind::CHM_SHORT:
				return createChmHnswInter<Coord, unsigned short>(type);
			case HnswKind::CHM_SIZE_T:
				return createChmHnswInter<Coord, size_t>(type);
			case HnswKind::HNSWLIB:
				return std::make_shared<HnswlibInterImpl<Coord>>(type->cfg);
			default:
				throw std::runtime_error("Unknown HnswKind value.");
		}
	}
}
