#pragma once
#include "Unique.hpp"

namespace chm {
	using IdxVec3D = std::vector<std::vector<std::vector<size_t>>>;
	using IdxVec3DPtr = std::shared_ptr<IdxVec3D>;

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

	inline HnswCfg::HnswCfg(
		const size_t dim, const size_t efConstruction, const size_t M, const size_t maxNodeCount, const unsigned int seed, const bool useEuclid
	) : dim(dim), efConstruction(efConstruction), M(M), maxNodeCount(maxNodeCount), seed(seed), useEuclid(useEuclid) {}
}
