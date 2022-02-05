#pragma once

namespace chm {
	template<typename Coord, typename Idx>
	struct Node {
		Coord dist;
		Idx idx;

		Node();
		Node(Coord dist, Idx idx);
	};

	template<typename Coord, typename Idx>
	struct FarCmp {
		using Node = Node<Coord, Idx>;

		constexpr bool operator()(const Node& a, const Node& b) const noexcept;
	};

	template<typename Coord, typename Idx>
	struct NearCmp {
		using Node = Node<Coord, Idx>;

		constexpr bool operator()(const Node& a, const Node& b) const noexcept;
	};

	template<typename Coord, typename Idx>
	inline Node<Coord, Idx>::Node() : dist(0), idx(0) {}

	template<typename Coord, typename Idx>
	inline Node<Coord, Idx>::Node(Coord dist, Idx idx) : dist(dist), idx(idx) {}

	template<typename Coord, typename Idx>
	inline constexpr bool FarCmp<Coord, Idx>::operator()(const Node& a, const Node& b) const noexcept {
		#ifdef DECIDE_BY_IDX

		if(a.dist == b.dist)
			return a.idx < b.idx;

		#endif
		return a.dist < b.dist;
	}

	template<typename Coord, typename Idx>
	inline constexpr bool NearCmp<Coord, Idx>::operator()(const Node& a, const Node& b) const noexcept {
		#ifdef DECIDE_BY_IDX

		if(a.dist == b.dist)
			return a.idx < b.idx;

		#endif

		return a.dist > b.dist;
	}
}
