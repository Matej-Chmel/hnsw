#pragma once
#include "BigNode.hpp"
#include <string>

namespace chm {
	enum class InterTest {
		ENTRY,
		FINAL_NEIGHBORS,
		LEVEL,
		LOWER_SEARCH_ANN,
		LOWER_SEARCH_ENTRY,
		LOWER_SEARCH_RES,
		SELECTED_NEIGHBORS,
		UPPER_SEARCH,
		UPPER_SEARCH_ANN
	};

	std::string str(const InterTest t);

	template<typename Coord>
	struct InterErr : public Unique {
		const size_t insertedIdx;
		const size_t lc;
		const BigNodeVecPtr<Coord> refNodes;
		const BigNodeVecPtr<Coord> subNodes;
		const InterTest test;

		InterErr(
			const size_t insertedIdx, const BigNodeVecPtr<Coord>& refNodes, const BigNodeVecPtr<Coord>& subNodes,
			const size_t lc, const InterTest test
		);
		void write(const fs::path& outDir) const;
	};

	template<typename Coord>
	using InterErrPtr = std::shared_ptr<InterErr<Coord>>;

	template<typename Coord>
	void writeVec(const BigNodeVecPtr<Coord>& vec, std::ostream& stream);

	template<typename Coord>
	void writeVec(const BigNodeVecPtr<Coord>& vec, const fs::path& path);

	template<typename Coord>
	inline InterErr<Coord>::InterErr(
		const size_t insertedIdx, const BigNodeVecPtr<Coord>& refNodes, const BigNodeVecPtr<Coord>& subNodes,
		const size_t lc, const InterTest test
	) : insertedIdx(insertedIdx), lc(lc), refNodes(refNodes), subNodes(subNodes), test(test) {}

	template<typename Coord>
	inline void InterErr<Coord>::write(const fs::path& outDir) const {
		writeVec(this->refNodes, outDir / "expected.log");
		writeVec(this->subNodes, outDir / "actual.log");

		std::ofstream s(outDir / "report.log");
		s << "Node: " << this->insertedIdx << '\n' << "Test: " << str(this->test) << ".\n" << "Layer: " << this->lc << '\n';
	}

	template<typename Coord>
	void writeVec(const BigNodeVecPtr<Coord>& vec, std::ostream& stream) {
		const auto& v = *vec;
		const auto len = v.size();

		stream << "[Length " << len << "]\n\n";

		for(size_t i = 0; i < len; i++) {
			const auto& node = v[i];
			stream << "[" << i << "]: " << node.idx << ", " << node.dist << '\n';
		}
	}

	template<typename Coord>
	void writeVec(const BigNodeVecPtr<Coord>& vec, const fs::path& path) {
		std::ofstream s(path);
		writeVec(vec, s);
	}
}
