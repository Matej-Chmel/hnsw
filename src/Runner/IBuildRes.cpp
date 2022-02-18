#include <stdexcept>
#include <sstream>
#include "IBuildRes.hpp"

namespace chm {
	template<typename T>
	LL convert(chr::microseconds& us) {
		const auto res = chr::duration_cast<T>(us);
		us -= chr::duration_cast<chr::microseconds>(res);
		return res.count();
	}

	std::string elapsedTimeToStr(const chr::microseconds& elapsed) {
		std::stringstream s;
		chr::microseconds us = elapsed;

		s << '[';
		padTimeNum(s, convert<chr::minutes>(us));
		s << ':';
		padTimeNum(s, convert<chr::seconds>(us));
		s << '.';
		padTimeNum(s, convert<chr::milliseconds>(us), 3);
		s << '.';
		padTimeNum(s, us.count(), 3);
		s << "] " << elapsed.count() << " us";

		return s.str();
	}

	void IRunRes::write(const fs::path&) const {
		throw std::runtime_error("Method IRunRes::write isn't overriden.");
	}

	void QueryTime::calcStats() {
		this->accumulated = std::accumulate(this->queries.cbegin(), this->queries.cend(), chr::microseconds(0));
		this->avg = chr::microseconds(this->accumulated / this->queries.size());
	}

	QueryTime::QueryTime(const size_t queryCount) : accumulated(0), avg(0) {
		this->queries.reserve(queryCount);
	}

	AlgoBuildRes::AlgoBuildRes(const size_t nodeCount) : conn(nullptr), init(0), queryTime(nodeCount) {}

	void AlgoBuildRes::print(std::ostream& s, const std::string& title) const {
		s << "\n[" << title << "]\n";
		printElapsedTime(s, "Initialization time", this->init);
		printElapsedTime(s, "Accumulated build time", this->queryTime.accumulated);
		printElapsedTime(s, "Average insert time", this->queryTime.avg);
	}

	bool IBuildRes::testLevels() const {
		const auto len = std::min(this->getRefRes().conn->size(), this->getSubRes().conn->size());

		for(size_t i = 0; i < len; i++)
			if((*this->getRefRes().conn)[i].size() != (*this->getSubRes().conn)[i].size())
				return false;

		return true;
	}

	bool IBuildRes::testNeighborLengths() const {
		const auto len = std::min(this->getRefRes().conn->size(), this->getSubRes().conn->size());

		for(size_t i = 0; i < len; i++) {
			const auto& nodeLayersRef = (*this->getRefRes().conn)[i];
			const auto& nodeLayersSub = (*this->getSubRes().conn)[i];
			const auto nodeLayersLen = std::min(nodeLayersRef.size(), nodeLayersSub.size());

			for(size_t layerIdx = 0; layerIdx < nodeLayersLen; layerIdx++)
				if(nodeLayersRef[layerIdx].size() != nodeLayersSub[layerIdx].size())
					return false;
		}

		return true;
	}

	bool IBuildRes::testNeighborIndices() const {
		const auto len = std::min(this->getRefRes().conn->size(), this->getSubRes().conn->size());

		for(size_t i = 0; i < len; i++) {
			const auto& nodeLayersRef = (*this->getRefRes().conn)[i];
			const auto& nodeLayersSub = (*this->getSubRes().conn)[i];
			const auto nodeLayersLen = std::min(nodeLayersRef.size(), nodeLayersSub.size());

			for(size_t layerIdx = 0; layerIdx < nodeLayersLen; layerIdx++) {
				const auto& neighborsRef = nodeLayersRef[layerIdx];
				const auto& neighborsSub = nodeLayersSub[layerIdx];
				const auto neighborsLen = std::min(neighborsRef.size(), neighborsSub.size());

				for(size_t neighborIdx = 0; neighborIdx < neighborsLen; neighborIdx++)
					if(neighborsRef[neighborIdx] != neighborsSub[neighborIdx])
						return false;
			}
		}

		return true;
	}

	bool IBuildRes::testNodeCount() const {
		return this->getRefRes().conn->size() == this->nodeCount && this->getSubRes().conn->size() == this->nodeCount;
	}

	IBuildRes::IBuildRes(const size_t nodeCount)
		: nodeCount(nodeCount), levelsPassed(false), neighborIndicesPassed(false), neighborLengthsPassed(false), nodeCountPassed(false) {}

	void IBuildRes::printTests(std::ostream& s) const {
		printTestRes(s, "Node count", this->nodeCountPassed);
		printTestRes(s, "Levels", this->levelsPassed);
		printTestRes(s, "Neighbors lengths", this->neighborLengthsPassed);
		printTestRes(s, "Neighbors indices", this->neighborIndicesPassed);
	}

	void IBuildRes::printTime(std::ostream& s) const {
		this->getRefRes().print(s, "Reference algorithm");
		this->getSubRes().print(s, "Subject algorithm");
	}

	void IBuildRes::print(std::ostream& s) const {
		this->printTests(s);
		this->printTime(s);
	}

	void IBuildRes::runTests() {
		this->levelsPassed = this->testLevels();
		this->neighborIndicesPassed = this->testNeighborIndices();
		this->neighborLengthsPassed = this->testNeighborLengths();
		this->nodeCountPassed = this->testNodeCount();
	}

	void IBuildRes::write(const fs::path& outDir) const {
		writeConn(this->getRefRes().conn, outDir / "refConn.log");
		writeConn(this->getSubRes().conn, outDir / "subConn.log");

		std::ofstream s(outDir / "build.log");
		this->print(s);
	}

	void printElapsedTime(std::ostream& s, const std::string& title, const chr::microseconds& elapsed) {
		s << title << ": " << elapsedTimeToStr(elapsed) << '\n';
	}

	void printTestRes(std::ostream& s, const std::string& title, const bool passed) {
		s << testResToStr(passed) << ' ' << title << " test.\n";
	}

	void SeqAlgoBuildRes::print(std::ostream& s, const std::string& title) const {
		AlgoBuildRes::print(s, title);
		printElapsedTime(s, "Total build time", this->total);
	}

	SeqAlgoBuildRes::SeqAlgoBuildRes(const size_t nodeCount) : AlgoBuildRes(nodeCount), total(0) {}

	const AlgoBuildRes& SeqBuildRes::getRefRes() const {
		return this->refRes;
	}

	const AlgoBuildRes& SeqBuildRes::getSubRes() const {
		return this->subRes;
	}

	SeqBuildRes::SeqBuildRes(const size_t nodeCount) : IBuildRes(nodeCount), refRes(nodeCount), subRes(nodeCount) {}

	std::string testResToStr(const bool passed) {
		std::stringstream s;
		s << '[' << (passed ? "PASSED" : "FAILED") << ']';
		return s.str();
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
}
