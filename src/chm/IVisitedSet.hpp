#pragma once
#include <unordered_set>
#include "Unique.hpp"

namespace chm {
	template<typename Idx>
	struct IVisitedSet : public Unique {
		virtual ~IVisitedSet() = default;
		virtual void prepare(const Idx count, const Idx entry) = 0;
		virtual bool insert(const Idx i) = 0;
	};

	template<typename Idx>
	using IVisitedSetPtr = std::shared_ptr<IVisitedSet<Idx>>;

	template<typename Idx>
	class VisitedBitset : public IVisitedSet<Idx> {
		std::vector<bool> v;

	public:
		void prepare(const Idx count, const Idx entry) override;
		bool insert(const Idx i) override;
	};

	template<typename Idx>
	class UnorderedVisitedSet : public IVisitedSet<Idx> {
		std::unordered_set<Idx> s;

	public:
		void prepare(const Idx count, const Idx entry) override;
		bool insert(const Idx i) override;
	};

	template<typename Idx>
	IVisitedSetPtr<Idx> createVisitedSet(const bool useBitset);

	template<typename Idx>
	inline void VisitedBitset<Idx>::prepare(const Idx count, const Idx entry) {
		this->v.clear();
		this->v.resize(count);
		this->v[entry] = true;
	}

	template<typename Idx>
	inline bool VisitedBitset<Idx>::insert(const Idx i) {
		if(this->v[i])
			return false;

		this->v[i] = true;
		return true;
	}

	template<typename Idx>
	inline void UnorderedVisitedSet<Idx>::prepare(const Idx count, const Idx entry) {
		this->s.clear();
		this->s.insert(entry);
	}

	template<typename Idx>
	inline bool UnorderedVisitedSet<Idx>::insert(const Idx i) {
		return this->s.insert(i).second;
	}

	template<typename Idx>
	IVisitedSetPtr<Idx> createVisitedSet(const bool useBitset) {
		if(useBitset)
			return std::make_shared<VisitedBitset<Idx>>();
		return std::make_shared<UnorderedVisitedSet<Idx>>();
	}
}
