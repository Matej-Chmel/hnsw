#pragma once
#include "IConnections.hpp"

namespace chm {
	template<typename Coord>
	struct IHnsw : public Unique {
		virtual ~IHnsw() = default;
		virtual IdxVec3DPtr getConnections() const = 0;
		virtual void insert(const ConstIter<Coord>& query) = 0;
		virtual void knnSearch(
			const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
		) = 0;
	};

	template<typename Coord>
	using IHnswPtr = std::shared_ptr<IHnsw<Coord>>;

	struct HnswCfg : public Unique {
		const size_t dim;
		const size_t efConstruction;
		const size_t M;
		const size_t maxNodeCount;
		const unsigned int seed;
		const bool useEuclid;

		HnswCfg(
			const size_t dim, const size_t efConstruction, const size_t M, const size_t maxNodeCount, const unsigned int seed, const bool useEuclid
		);
	};

	using HnswCfgPtr = std::shared_ptr<HnswCfg>;

	struct HnswSettings : public Unique {
		const bool distanceCacheEnabled;
		const bool useBitset;
		const bool useConnLayer0;
		
		HnswSettings(const bool distanceCacheEnabled, const bool useBitset, const bool useConnLayer0);
	};

	using HnswSettingsPtr = std::shared_ptr<HnswSettings>;

	inline HnswCfg::HnswCfg(
		const size_t dim, const size_t efConstruction, const size_t M, const size_t maxNodeCount, const unsigned int seed, const bool useEuclid
	) : dim(dim), efConstruction(efConstruction), M(M), maxNodeCount(maxNodeCount), seed(seed), useEuclid(useEuclid) {}

	inline HnswSettings::HnswSettings(const bool distanceCacheEnabled, const bool useBitset, const bool useConnLayer0)
		: distanceCacheEnabled(distanceCacheEnabled), useBitset(useBitset), useConnLayer0(useConnLayer0) {}
}
