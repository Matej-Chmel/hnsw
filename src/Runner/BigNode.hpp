#pragma once
#include "chm/Hnsw.hpp"
#include "common.hpp"

namespace chm {
	template<typename Coord>
	using BigNode = Node<Coord, size_t>;

	template<typename Coord>
	using BigNodeVec = std::vector<BigNode<Coord>>;

	template<typename Coord>
	using BigNodeVecPtr = VecPtr<BigNode<Coord>>;

	template<typename Coord>
	struct BigNodeCmp {
		constexpr bool operator()(const BigNode<Coord>& a, const BigNode<Coord>& b) const noexcept;
	};

	template<typename Coord, typename Idx>
	BigNodeVecPtr<Coord> copyToVec(const FarHeap<Coord, Idx>& h);

	struct LevelRng : public Unique {
		const size_t start;
		const size_t end;

		LevelRng(const size_t start, const size_t end);
	};

	using LevelRngPtr = std::shared_ptr<LevelRng>;

	template<typename Coord>
	inline constexpr bool BigNodeCmp<Coord>::operator()(const BigNode<Coord>& a, const BigNode<Coord>& b) const noexcept {
		if(a.dist == b.dist)
			return a.idx < b.idx;
		return a.dist < b.dist;
	}

	template<typename Coord, typename Idx>
	BigNodeVecPtr<Coord> copyToVec(const FarHeap<Coord, Idx>& h) {
		auto res = std::make_shared<BigNodeVec<Coord>>();
		res->reserve(h.len());

		for(const auto& n : h)
			res->push_back(Node(n.dist, size_t(n.idx)));

		return res;
	}
}
