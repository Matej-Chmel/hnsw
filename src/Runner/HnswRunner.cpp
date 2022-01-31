#include "HnswRunner.hpp"

namespace chm {
	template<typename T>
	LL convert(chr::microseconds& us) {
		const auto res = chr::duration_cast<T>(us);
		us -= chr::duration_cast<chr::microseconds>(res);
		return res.count();
	}

	void printElapsedTime(const std::string& title, const chr::microseconds& elapsed, std::ostream& s) {
		chr::microseconds us = elapsed;

		s << title << " time: [";
		padTimeNum(s, convert<chr::minutes>(us));
		s << ':';
		padTimeNum(s, convert<chr::seconds>(us));
		s << '.';
		padTimeNum(s, convert<chr::milliseconds>(us), 3);
		s << '.';
		padTimeNum(s, us.count(), 3);
		s << "] " << elapsed.count() << " us\n";
	}

	std::string str(const InterTest t) {
		switch(t) {
			case InterTest::ENTRY:
				return "Graph entry point";
			case InterTest::FINAL_NEIGHBORS:
				return "Neighbors after connecting";
			case InterTest::LEVEL:
				return "Last generated level";
			case InterTest::LOWER_SEARCH_ENTRY:
				return "Entry point of lower search";
			case InterTest::LOWER_SEARCH_RES:
				return "Results of lower search";
			case InterTest::SELECTED_NEIGHBORS:
				return "Original selected neighbors";
			case InterTest::UPPER_SEARCH:
				return "Nearest node from upper search";
			default:
				throw std::runtime_error("Unknown InterTest value.");
		}
	}

	AccBuildTime::AccBuildTime(const size_t nodeCount) : accumulated(0), avgInsert(0), init(0) {
		this->inserts.reserve(nodeCount);
	}

	void AccBuildTime::calcStats() {
		const auto sum = std::accumulate(this->inserts.cbegin(), this->inserts.cend(), chr::microseconds(0));
		this->accumulated = this->init + sum;
		this->avgInsert = chr::microseconds(sum / this->inserts.size());
	}

	void AccBuildTime::print(const std::string& name, std::ostream& s) const {
		s << "\n[" << name << "]\n";
		printElapsedTime("Average insert", this->avgInsert, s);
		printElapsedTime("Initialization", this->init, s);
		printElapsedTime("Accumulated build", this->accumulated, s);
	}

	TotalBuildTime::TotalBuildTime(const size_t nodeCount) : AccBuildTime(nodeCount), total(0) {}

	void TotalBuildTime::print(const std::string& name, std::ostream& s) const {
		AccBuildTime::print(name, s);
		printElapsedTime("Total build", this->total, s);
	}

	HnswRunCfg::HnswRunCfg(const HnswTypePtr& refType, const HnswTypePtr& subType) : refType(refType), subType(subType) {}

	void Timer::start() {
		this->from = chr::steady_clock::now();
	}

	chr::microseconds Timer::stop() {
		return chr::duration_cast<chr::microseconds>(chr::steady_clock::now() - this->from);
	}

	Timer::Timer() : from{} {}

	void printTestRes(const std::string& title, const bool passed, std::ostream& s) {
		s << '[' << (passed ? "PASSED" : "FAILED") << "] " << title << " test.\n";
	}

	IdxVec3DPtr sortedInPlace(const IdxVec3DPtr& conn) {
		for(auto& nodeLayers : *conn)
			for(auto& layer : nodeLayers)
				std::sort(layer.begin(), layer.end());
		return conn;
	}

	void writeConn(const IdxVec3DPtr& conn, std::ostream& stream) {
		const auto& c = *conn;
		const auto nodeCount = c.size();
		const auto nodeLastIdx = nodeCount - 1;

		for(size_t nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++) {
			stream << "Node " << nodeIdx << '\n';

			const auto& nodeLayers = c[nodeIdx];
			const auto nodeLayersLen = nodeLayers.size();

			for(size_t layerIdx = nodeLayersLen - 1;; layerIdx--) {
				stream << "Layer " << layerIdx << ": ";

				const auto& layer = nodeLayers[layerIdx];

				if(layer.empty()) {
					stream << "EMPTY\n";
				} else {
					const auto layerLen = layer.size();
					const auto lastIdx = layerLen - 1;

					stream << "[length " << layerLen << "] ";

					for(size_t i = 0; i < lastIdx; i++)
						stream << layer[i] << ' ';

					stream << layer[lastIdx] << '\n';
				}

				if(!layerIdx)
					break;
			}

			if(nodeIdx != nodeLastIdx)
				stream << '\n';
		}
	}

	void writeConn(const IdxVec3DPtr& conn, const fs::path& path) {
		std::ofstream s(path);
		writeConn(conn, s);
	}

	HnswType::HnswType(const bool isIntermediate, const HnswKind kind) : isIntermediate(isIntermediate), kind(kind) {}
}
