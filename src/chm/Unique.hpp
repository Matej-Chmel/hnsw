#pragma once
#include <memory>
#include <vector>

namespace chm {
	template<typename T>
	using ConstIter = typename std::vector<T>::const_iterator;

	template<typename T>
	using Iter = typename std::vector<T>::iterator;

	class Unique {
	protected:
		Unique() = default;

	public:
		Unique& operator=(const Unique&) = delete;
		Unique& operator=(Unique&&) = delete;
		Unique(const Unique&) = delete;
		Unique(Unique&&) = delete;
	};
}
