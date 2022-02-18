#pragma once
#include <iomanip>
#include "IBuildRes.hpp"

namespace chm {
	constexpr std::streamsize ELAPSED_W = 36;
	constexpr std::streamsize RECALL_W = 20;
	constexpr std::streamsize TEST_W = 26;

	using Indices2D = std::vector<std::vector<size_t>>;

	template<typename Coord>
	struct FoundNeighbors : public Unique {
		std::vector<std::vector<Coord>> distances;
		Indices2D indices;

		FoundNeighbors(const size_t queryCount);
	};

	template<typename Coord>
	struct AlgoSearchRes : public Unique {
		FoundNeighbors<Coord> neighbors;
		QueryTime queryTime;
		float recall;

		virtual ~AlgoSearchRes() = default;
		AlgoSearchRes(const size_t queryCount);
		void calcRecall(Indices2D& correctIndices);
		virtual void print(std::ostream& s, const std::string& title) const;
	};

	template<typename Coord>
	class ISearchRes : public IRunRes {
		const size_t queryCount;

		bool testNeighborLengths() const;
		bool testNeighborIndices() const;
		bool testQueryCount() const;

	protected:
		virtual const AlgoSearchRes<Coord>& getRefRes() const = 0;
		virtual const AlgoSearchRes<Coord>& getSubRes() const = 0;
		ISearchRes(const size_t queryCount);
		void printAlgoRes(std::ostream& s) const;
		void printTests(std::ostream& s) const;

	public:
		bool neighborIndicesPassed;
		bool neighborLengthsPassed;
		bool queryCountPassed;

		virtual ~ISearchRes() = default;
		virtual void print(std::ostream& s) const override;
		void runTests() override;
	};

	template<typename Coord>
	using ISearchResPtr = std::shared_ptr<ISearchRes<Coord>>;

	template<typename Coord>
	class InterSearchRes : public ISearchRes<Coord> {
	protected:
		const AlgoSearchRes<Coord>& getRefRes() const override;
		const AlgoSearchRes<Coord>& getSubRes() const override;

	public:
		InterErrPtr<Coord> err;
		AlgoSearchRes<Coord> refRes;
		AlgoSearchRes<Coord> subRes;

		InterSearchRes(const size_t queryCount);
		void print(std::ostream& s) const override;
		void writeErr(const fs::path& outDir) const;
	};

	template<typename Coord>
	using InterSearchResPtr = std::shared_ptr<InterSearchRes<Coord>>;

	template<typename Coord>
	struct SeqAlgoSearchRes : public AlgoSearchRes<Coord> {
		chr::microseconds total;

		void print(std::ostream& s, const std::string& title) const;
		SeqAlgoSearchRes(const size_t queryCount);
	};

	template<typename Coord>
	class SeqSearchRes : public ISearchRes<Coord> {
	protected:
		const AlgoSearchRes<Coord>& getRefRes() const override;
		const AlgoSearchRes<Coord>& getSubRes() const override;

	public:
		SeqAlgoSearchRes<Coord> refRes;
		SeqAlgoSearchRes<Coord> subRes;

		SeqSearchRes(const size_t queryCount);
	};

	template<typename Coord>
	using SeqSearchResPtr = std::shared_ptr<SeqSearchRes<Coord>>;

	template<typename Coord>
	inline FoundNeighbors<Coord>::FoundNeighbors(const size_t queryCount) {
		this->distances.resize(queryCount);
		this->indices.resize(queryCount);
	}

	template<typename Coord>
	inline AlgoSearchRes<Coord>::AlgoSearchRes(const size_t queryCount) : neighbors(queryCount), queryTime(queryCount), recall(0.f) { }

	template<typename Coord>
	inline void AlgoSearchRes<Coord>::calcRecall(Indices2D& correctIndices) {
		size_t correct = 0;
		const auto queryCount = std::min(this->neighbors.indices.size(), correctIndices.size());
		size_t total = 0;

		for(size_t i = 0; i < queryCount; i++) {
			const std::unordered_set<size_t> correctSet(correctIndices[i].cbegin(), correctIndices[i].cend());
			total += correctSet.size();

			for(const auto& found : this->neighbors.indices[i])
				if(correctSet.find(found) != correctSet.end())
					correct++;
		}

		this->recall = 1.f * correct / total;
	}

	template<typename Coord>
	inline void AlgoSearchRes<Coord>::print(std::ostream& s, const std::string& title) const {
		s << "\n[" << title << "]\nRecall: " << this->recall << '\n';
		printElapsedTime(s, "Accumulated search time", this->queryTime.accumulated);
		printElapsedTime(s, "Average query time", this->queryTime.avg);
	}

	template<typename Coord>
	inline bool ISearchRes<Coord>::testNeighborLengths() const {
		const auto& refIndices = this->getRefRes().neighbors.indices;
		const auto& subIndices = this->getSubRes().neighbors.indices;
		const auto queryCount = std::min(refIndices.size(), subIndices.size());

		for(size_t i = 0; i < queryCount; i++)
			if(refIndices[i].size() != subIndices[i].size())
				return false;

		return true;
	}

	template<typename Coord>
	inline bool ISearchRes<Coord>::testNeighborIndices() const {
		const auto& refIndices = this->getRefRes().neighbors.indices;
		const auto& subIndices = this->getSubRes().neighbors.indices;
		const auto queryCount = std::min(refIndices.size(), subIndices.size());

		for(size_t i = 0; i < queryCount; i++) {
			std::unordered_set<size_t> queryRefIndices(refIndices[i].cbegin(), refIndices[i].cend());
			const auto& querySubIndices = subIndices[i];

			for(const auto& neighbor : querySubIndices)
				if(queryRefIndices.find(neighbor) == queryRefIndices.end())
					return false;
		}

		return true;
	}

	template<typename Coord>
	inline bool ISearchRes<Coord>::testQueryCount() const {
		return this->getRefRes().neighbors.indices.size() == this->queryCount && this->getSubRes().neighbors.indices.size() == this->queryCount;
	}

	template<typename Coord>
	inline ISearchRes<Coord>::ISearchRes(const size_t queryCount)
		: queryCount(queryCount), neighborIndicesPassed(false), neighborLengthsPassed(false), queryCountPassed(false) {}

	template<typename Coord>
	inline void ISearchRes<Coord>::printAlgoRes(std::ostream& s) const {
		this->getRefRes().print(s, "Reference algorithm");
		this->getSubRes().print(s, "Subject algorithm");
	}

	template<typename Coord>
	inline void ISearchRes<Coord>::printTests(std::ostream& s) const {
		printTestRes(s, "Query count", this->queryCountPassed);
		printTestRes(s, "Results lengths", this->neighborLengthsPassed);
		printTestRes(s, "Results indices", this->neighborIndicesPassed);
	}

	template<typename Coord>
	inline void ISearchRes<Coord>::print(std::ostream& s) const {
		this->printTests(s);
		this->printAlgoRes(s);
	}

	template<typename Coord>
	inline void ISearchRes<Coord>::runTests() {
		this->neighborIndicesPassed = this->testNeighborIndices();
		this->neighborLengthsPassed = this->testNeighborLengths();
		this->queryCountPassed = this->testQueryCount();
	}

	template<typename Coord>
	inline const AlgoSearchRes<Coord>& InterSearchRes<Coord>::getRefRes() const {
		return this->refRes;
	}

	template<typename Coord>
	inline const AlgoSearchRes<Coord>& InterSearchRes<Coord>::getSubRes() const {
		return this->subRes;
	}

	template<typename Coord>
	inline InterSearchRes<Coord>::InterSearchRes(const size_t queryCount)
		: ISearchRes<Coord>(queryCount), err(nullptr), refRes(queryCount), subRes(queryCount) {}

	template<typename Coord>
	inline void InterSearchRes<Coord>::print(std::ostream& s) const {
		this->printTests(s);
		printTestRes(s, "Intermediates comparison", !this->err);
		this->printAlgoRes(s);
	}

	template<typename Coord>
	inline void InterSearchRes<Coord>::writeErr(const fs::path& outDir) const {
		if(this->err) {
			if(!fs::exists(outDir))
				fs::create_directories(outDir);
			this->err->write(outDir);
		}
	}

	template<typename Coord>
	inline void SeqAlgoSearchRes<Coord>::print(std::ostream& s, const std::string& title) const {
		AlgoSearchRes<Coord>::print(s, title);
		printElapsedTime(s, "Total search time", this->total);
	}

	template<typename Coord>
	inline SeqAlgoSearchRes<Coord>::SeqAlgoSearchRes(const size_t queryCount) : AlgoSearchRes<Coord>(queryCount), total(0) {}

	template<typename Coord>
	inline const AlgoSearchRes<Coord>& SeqSearchRes<Coord>::getRefRes() const {
		return this->refRes;
	}

	template<typename Coord>
	inline const AlgoSearchRes<Coord>& SeqSearchRes<Coord>::getSubRes() const {
		return this->subRes;
	}

	template<typename Coord>
	inline SeqSearchRes<Coord>::SeqSearchRes(const size_t queryCount) : ISearchRes<Coord>(queryCount), refRes(queryCount), subRes(queryCount) {}
}
