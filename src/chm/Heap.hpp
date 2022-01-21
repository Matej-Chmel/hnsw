#pragma once
#include <algorithm>
#include <vector>

namespace chm {
	template<typename Coord, typename Idx>
	struct Node {
		Coord dist;
		Idx idx;

		Node();
		Node(Coord dist, Idx idx);
	};

	template<typename Coord, typename Idx>
	struct FarComparator {
		using Node = Node<Coord, Idx>;

		constexpr bool operator()(const Node& a, const Node& b) const noexcept;
	};

	template<typename Coord, typename Idx>
	struct NearComparator {
		using Node = Node<Coord, Idx>;

		constexpr bool operator()(const Node& a, const Node& b) const noexcept;
	};

	template<typename Coord, typename Idx, class Comparator>
	class Heap {
	public:
		using Node = Node<Coord, Idx>;
		using IdxVec = std::vector<Idx>;

	private:
		using NodeVec = std::vector<Node>;
		typedef typename NodeVec::iterator iterator;

		NodeVec nodes;

		iterator begin();
		iterator end();

	public:
		typedef typename NodeVec::const_iterator const_iterator;

		const_iterator begin() const noexcept;
		void clear();
		const_iterator end() const noexcept;
		void extractTo(IdxVec& v);
		Heap() = default;
		Heap(const Node& ep);
		template<class C> Heap(Heap<Coord, Idx, C>& o);
		size_t len() const;
		void pop();
		void push(const Node& n);
		void push(const Coord dist, const Idx idx);
		void reserve(size_t capacity);
		const Node& top();
	};

	template<typename Coord, typename Idx>
	using FarHeap = Heap<Coord, Idx, FarComparator<Coord, Idx>>;

	template<typename Coord, typename Idx>
	using NearHeap = Heap<Coord, Idx, NearComparator<Coord, Idx>>;

	template<typename Coord, typename Idx>
	inline Node<Coord, Idx>::Node() : dist(0), idx(0) {}

	template<typename Coord, typename Idx>
	inline Node<Coord, Idx>::Node(Coord dist, Idx idx) : dist(dist), idx(idx) {}

	template<typename Coord, typename Idx>
	inline constexpr bool FarComparator<Coord, Idx>::operator()(const Node& a, const Node& b) const noexcept {
		return a.dist < b.dist;
	}

	template<typename Coord, typename Idx>
	inline constexpr bool NearComparator<Coord, Idx>::operator()(const Node& a, const Node& b) const noexcept {
		return a.dist > b.dist;
	}

	template<typename Coord, typename Idx, class Comparator>
	inline typename Heap<Coord, Idx, Comparator>::iterator Heap<Coord, Idx, Comparator>::begin() {
		return this->nodes.begin();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline typename Heap<Coord, Idx, Comparator>::iterator Heap<Coord, Idx, Comparator>::end() {
		return this->nodes.end();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline typename Heap<Coord, Idx, Comparator>::const_iterator Heap<Coord, Idx, Comparator>::begin() const noexcept {
		return this->nodes.begin();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::clear() {
		this->nodes.clear();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline typename Heap<Coord, Idx, Comparator>::const_iterator Heap<Coord, Idx, Comparator>::end() const noexcept {
		return this->nodes.end();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::extractTo(IdxVec& v) {
		v.clear();
		v.reserve(this->len());

		while(this->len() > 1) {
			v.emplace_back(this->top().idx);
			this->pop();
		}

		v.emplace_back(this->top().idx);
	}

	template<typename Coord, typename Idx, class Comparator>
	inline Heap<Coord, Idx, Comparator>::Heap(const Node& ep) {
		this->push(ep);
	}

	template<typename Coord, typename Idx, class Comparator>
	template<class C>
	inline Heap<Coord, Idx, Comparator>::Heap(Heap<Coord, Idx, C>& o) {
		this->reserve(o.len());

		while(o.len()) {
			this->push(o.top());
			o.pop();
		}
	}

	template<typename Coord, typename Idx, class Comparator>
	inline size_t Heap<Coord, Idx, Comparator>::len() const {
		return this->nodes.size();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::pop() {
		std::pop_heap(this->begin(), this->end(), Comparator());
		this->nodes.pop_back();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::push(const Node& n) {
		this->push(n.dist, n.idx);
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::push(const Coord dist, const Idx idx) {
		this->nodes.emplace_back(dist, idx);
		std::push_heap(this->begin(), this->end(), Comparator());
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::reserve(size_t capacity) {
		this->nodes.reserve(capacity);
	}

	template<typename Coord, typename Idx, class Comparator>
	inline const typename Heap<Coord, Idx, Comparator>::Node& Heap<Coord, Idx, Comparator>::top() {
		return this->nodes.front();
	}
}
