#include "HnswRunner.hpp"

namespace chm {
	template<typename T>
	void printElapsedMS(const std::string& title, const T elapsedMS, std::ostream& s) {
		s << title << " time: " << elapsedMS << " ms\n";
	}

	void printTestRes(const std::string& title, const bool passed, std::ostream& s) {
		s << '[' << (passed ? "PASSED" : "FAILED") << "] " << title << " test.\n";
	}

	BuildTimeRes::BuildTimeRes(const size_t nodeCount) : avgInsertMS(0.0), initMS(0), totalMS(0) {
		this->insertMS.reserve(nodeCount);
	}

	void BuildTimeRes::calcStats() {
		const auto sumInsertMS = std::accumulate(this->insertMS.cbegin(), this->insertMS.cend(), LL(0));

		this->avgInsertMS = double(sumInsertMS) / double(this->insertMS.size());
		this->totalMS = this->initMS + sumInsertMS;
	}

	void BuildTimeRes::print(const std::string& name, std::ostream& s) const {
		s << "\n[" << name << "]\n";
		printElapsedMS("Average insert", this->avgInsertMS, s);
		printElapsedMS("Initialization", this->initMS, s);
		printElapsedMS("Total build", this->totalMS, s);
	}

	HnswRunCfg::HnswRunCfg(const HnswTypePtr& refType, const HnswTypePtr& subType) : refType(refType), subType(subType) {}

	void Timer::start() {
		this->from = chr::system_clock::now();
	}

	LL Timer::stopMS() {
		return chr::duration_cast<chr::milliseconds>(chr::system_clock::now() - this->from).count();
	}

	Timer::Timer() : from{} {}

	IdxVec3DPtr sortedInPlace(const IdxVec3DPtr& conn) {
		for(auto& nodeLayers : *conn)
			for(auto& layer : nodeLayers)
				std::sort(layer.begin(), layer.end());
		return conn;
	}

	HnswType::HnswType(const bool isIntermediate, const HnswKind kind) : isIntermediate(isIntermediate), kind(kind) {}
}
