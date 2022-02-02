#pragma once
#include "hnswFactoryMethods.hpp"
#include "ICoords.hpp"

namespace chm {
	struct HnswRunCfg : public Unique {
		const HnswTypePtr refType;
		const HnswTypePtr subType;

		HnswRunCfg(const HnswTypePtr& refType, const HnswTypePtr& subType);
	};

	using HnswRunCfgPtr = std::shared_ptr<HnswRunCfg>;

	template<typename Coord>
	struct SearchCfg : public Unique {
		ICoordsPtr<Coord> coords;
		const size_t ef;
		const size_t K;

		void print(std::ostream& s) const;
		SearchCfg(const ICoordsPtr<Coord>& coords, const size_t ef, const size_t K);
	};

	template<typename Coord>
	using SearchCfgPtr = std::shared_ptr<SearchCfg<Coord>>;

	IdxVec3DPtr sortedInPlace(const IdxVec3DPtr& conn);

	class Timer : public Unique {
		chr::steady_clock::time_point from;

	public:
		void start();
		chr::microseconds stop();
		Timer();
	};

	template<typename Coord>
	inline void SearchCfg<Coord>::print(std::ostream& s) const {
		s << "EF: " << this->ef << "\nK: " << this->K << '\n';
	}

	template<typename Coord>
	inline SearchCfg<Coord>::SearchCfg(const ICoordsPtr<Coord>& coords, const size_t ef, const size_t K) : coords(coords), ef(ef), K(K) {}
}
