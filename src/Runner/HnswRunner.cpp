#include "HnswRunner.hpp"

namespace chm {
	void printElapsedMS(const std::string& title, const LL elapsedMS, std::ostream& s) {
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

	void BuildTimeRes::print(const std::string& name, std::ostream& s) {
		s << "\n[" << name << "]\n";
		printElapsedMS("Average insert", this->avgInsertMS, s);
		printElapsedMS("Initialization", this->initMS, s);
		printElapsedMS("Total build", this->totalMS, s);
	}

	void RunRes::print(std::ostream& s) {
		printTestRes("Node count", this->nodeCountPassed, s);
		printTestRes("Levels", this->levelsPassed, s);
		printTestRes("Neighbors lengths", this->neighborLengthsPassed, s);
		printTestRes("Neighbors indices", this->neighborIndicesPassed, s);
		this->refBuild.print("Reference algorithm", s);
		this->subBuild.print("Subject algorithm", s);
	}

	RunRes::RunRes(const size_t nodeCount)
		: levelsPassed(false), neighborIndicesPassed(false), neighborLengthsPassed(false), nodeCountPassed(false),
		refBuild(nodeCount), subBuild(nodeCount) {
	}

	HnswRunCfg::HnswRunCfg(const HnswKind refKind, const HnswKind subKind) : refKind(refKind), subKind(subKind) {}

	void Timer::start() {
		this->from = chr::system_clock::now();
	}

	LL Timer::stopMS() {
		return chr::duration_cast<chr::milliseconds>(chr::system_clock::now() - this->from).count();
	}

	Timer::Timer() : from{} {}
}
