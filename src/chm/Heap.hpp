#pragma once
#include <algorithm>
#include <vector>

namespace chm {
	class Unique {
	protected:
		Unique() = default;

	public:
		Unique& operator=(const Unique&) = delete;
		Unique& operator=(Unique&&) = delete;
		Unique(const Unique&) = delete;
		Unique(Unique&&) = delete;
	};

	using Idx = unsigned int;
	using IdxVec = std::vector<Idx>;

	template<typename Dist>
	struct Node {
		Dist dist;
		Idx idx;

		Node();
		Node(const Dist dist, const Idx idx);
	};

	template<typename Dist>
	struct FarCmp {
		constexpr bool operator()(const Node<Dist>& a, const Node<Dist>& b) const noexcept;
	};

	template<typename Dist>
	struct NearCmp {
		constexpr bool operator()(const Node<Dist>& a, const Node<Dist>& b) const noexcept;
	};

	template<typename T>
	using ConstIter = typename std::vector<T>::const_iterator;

	template<typename T>
	using Iter = typename std::vector<T>::iterator;

	template<typename Dist, class Cmp>
	class Heap {
		std::vector<Node<Dist>> nodes;

		Iter<Node<Dist>> begin();
		Iter<Node<Dist>> end();

	public:
		ConstIter<Node<Dist>> begin() const noexcept;
		void clear();
		ConstIter<Node<Dist>> end() const noexcept;
		void extractTo(IdxVec& v);
		Heap() = default;
		Heap(const Node<Dist>& ep);
		template<class C> Heap(Heap<Dist, C>& o);
		size_t len() const;
		template<class C> void loadFrom(Heap<Dist, C>& o);
		void pop();
		void push(const Node<Dist>& n);
		void push(const Dist dist, const Idx idx);
		void reserve(const size_t capacity);
		const Node<Dist>& top();
	};

	template<typename Dist>
	using FarHeap = Heap<Dist, FarCmp<Dist>>;

	template<typename Dist>
	using NearHeap = Heap<Dist, NearCmp<Dist>>;

	template<typename Dist>
	inline Node<Dist>::Node() : dist(0), idx(0) {}

	template<typename Dist>
	inline Node<Dist>::Node(const Dist dist, const Idx idx) : dist(dist), idx(idx) {}

	template<typename Dist>
	inline constexpr bool FarCmp<Dist>::operator()(const Node<Dist>& a, const Node<Dist>& b) const noexcept {
		#ifdef DECIDE_BY_IDX

		if(a.dist == b.dist)
			return a.idx < b.idx;

		#endif
		return a.dist < b.dist;
	}

	template<typename Dist>
	inline constexpr bool NearCmp<Dist>::operator()(const Node<Dist>& a, const Node<Dist>& b) const noexcept {
		#ifdef DECIDE_BY_IDX

		if(a.dist == b.dist)
			return a.idx < b.idx;

		#endif

		return a.dist > b.dist;
	}

	template<typename Dist, class Cmp>
	inline Iter<Node<Dist>> Heap<Dist, Cmp>::begin() {
		return this->nodes.begin();
	}

	template<typename Dist, class Cmp>
	inline Iter<Node<Dist>> Heap<Dist, Cmp>::end() {
		return this->nodes.end();
	}

	template<typename Dist, class Cmp>
	inline ConstIter<Node<Dist>> Heap<Dist, Cmp>::begin() const noexcept {
		return this->nodes.cbegin();
	}

	template<typename Dist, class Cmp>
	inline void Heap<Dist, Cmp>::clear() {
		this->nodes.clear();
	}

	template<typename Dist, class Cmp>
	inline ConstIter<Node<Dist>> Heap<Dist, Cmp>::end() const noexcept {
		return this->nodes.cend();
	}

	template<typename Dist, class Cmp>
	inline void Heap<Dist, Cmp>::extractTo(IdxVec& v) {
		v.clear();
		v.reserve(this->len());

		while(this->len() > 1) {
			v.emplace_back(this->top().idx);
			this->pop();
		}

		v.emplace_back(this->top().idx);
	}

	template<typename Dist, class Cmp>
	inline Heap<Dist, Cmp>::Heap(const Node<Dist>& ep) {
		this->push(ep);
	}

	template<typename Dist, class Cmp>
	template<class C>
	inline Heap<Dist, Cmp>::Heap(Heap<Dist, C>& o) {
		this->clear();
		this->reserve(o.len());

		while(o.len()) {
			this->push(o.top());
			o.pop();
		}
	}

	template<typename Dist, class Cmp>
	template<class C>
	inline void Heap<Dist, Cmp>::loadFrom(Heap<Dist, C>& o) {
		this->clear();

		while(o.len()) {
			this->push(o.top());
			o.pop();
		}
	}

	template<typename Dist, class Cmp>
	inline size_t Heap<Dist, Cmp>::len() const {
		return this->nodes.size();
	}

	template<typename Dist, class Cmp>
	inline void Heap<Dist, Cmp>::pop() {
		std::pop_heap(this->begin(), this->end(), Cmp());
		this->nodes.pop_back();
	}

	template<typename Dist, class Cmp>
	inline void Heap<Dist, Cmp>::push(const Node<Dist>& n) {
		this->push(n.dist, n.idx);
	}

	template<typename Dist, class Cmp>
	inline void Heap<Dist, Cmp>::push(const Dist dist, const Idx idx) {
		this->nodes.emplace_back(dist, idx);
		std::push_heap(this->begin(), this->end(), Cmp());
	}

	template<typename Dist, class Cmp>
	inline void Heap<Dist, Cmp>::reserve(const size_t capacity) {
		this->nodes.reserve(capacity);
	}

	template<typename Dist, class Cmp>
	inline const Node<Dist>& Heap<Dist, Cmp>::top() {
		return this->nodes.front();
	}
}
