#include "HnswRunner.hpp"

namespace chm {
	template<typename T>
	void printElapsedTime(const std::string& title, const T elapsedMS, std::ostream& s) {
		s << title << " time: " << elapsedMS << " ms [";
		padTimeNum(s, LL(elapsedMS) / 3600000LL);

		auto remainder = LL(elapsedMS) % 3600000LL;
		s << ':';
		padTimeNum(s, remainder / 60000LL);

		remainder %= 60000LL;
		s << ':';
		padTimeNum(s, remainder / 1000LL);
		s << '.';
		padTimeNum(s, remainder % 1000LL, 3);
		s << "]\n";
	}

	const char* const str(const InterTest t) {
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
				return nullptr;
		}
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
		printElapsedTime("Average insert", this->avgInsertMS, s);
		printElapsedTime("Initialization", this->initMS, s);
		printElapsedTime("Total build", this->totalMS, s);
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
