#pragma once
#include <algorithm>
#include "Node.hpp"
#include "IConnections.hpp"

namespace chm {
	template<typename Coord, typename Idx, class Comparator>
	class Heap : public Unique {
	public:
		using Node = Node<Coord, Idx>;

	private:
		std::vector<Node> nodes;

		Iter<Node> begin();
		Iter<Node> end();

	public:
		ConstIter<Node> begin() const noexcept;
		void clear();
		ConstIter<Node> end() const noexcept;
		void extractTo(const INeighborsPtr<Idx>& n);
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
	using FarHeap = Heap<Coord, Idx, FarCmp<Coord, Idx>>;

	template<typename Coord, typename Idx>
	using NearHeap = Heap<Coord, Idx, NearCmp<Coord, Idx>>;

	template<typename Coord, typename Idx, class Comparator>
	inline Iter<Node<Coord, Idx>> Heap<Coord, Idx, Comparator>::begin() {
		return this->nodes.begin();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline Iter<Node<Coord, Idx>> Heap<Coord, Idx, Comparator>::end() {
		return this->nodes.end();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline ConstIter<Node<Coord, Idx>> Heap<Coord, Idx, Comparator>::begin() const noexcept {
		return this->nodes.begin();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::clear() {
		this->nodes.clear();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline ConstIter<Node<Coord, Idx>> Heap<Coord, Idx, Comparator>::end() const noexcept {
		return this->nodes.end();
	}

	template<typename Coord, typename Idx, class Comparator>
	inline void Heap<Coord, Idx, Comparator>::extractTo(const INeighborsPtr<Idx>& n) {
		n->clear();
		n->reserve(this->len());

		while(this->len() > 1) {
			n->push(this->top().idx);
			this->pop();
		}

		n->push(this->top().idx);
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
