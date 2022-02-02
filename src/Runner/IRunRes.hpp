#pragma once
#include <fstream>
#include <numeric>
#include <ostream>
#include <string>
#include "chm/Hnsw.hpp"
#include "common.hpp"
#include "InterErr.hpp"

namespace chm {
	struct IRunRes : public Unique {
		virtual void print(std::ostream& s) const = 0;
		virtual void runTests() = 0;
		virtual void write(const fs::path& outDir) const = 0;
	};

	struct QueryTime : public Unique {
		chr::microseconds accumulated;
		chr::microseconds avg;
		std::vector<chr::microseconds> queries;

		void calcStats();
		QueryTime(const size_t queryCount);
	};

	struct AlgoBuildRes : public Unique {
		IdxVec3DPtr conn;
		chr::microseconds init;
		QueryTime queryTime;

		AlgoBuildRes(const size_t nodeCount);
		virtual void print(std::ostream& s, const std::string& title) const;
	};

	class IBuildRes : public IRunRes {
		size_t nodeCount;

		bool testLevels() const;
		bool testNeighborLengths() const;
		bool testNeighborIndices() const;
		bool testNodeCount() const;

	protected:
		IBuildRes(const size_t nodeCount);
		void printTests(std::ostream& s) const;
		void printTime(std::ostream& s) const;
		virtual const AlgoBuildRes& getRefRes() const = 0;
		virtual const AlgoBuildRes& getSubRes() const = 0;

	public:
		bool levelsPassed;
		bool neighborIndicesPassed;
		bool neighborLengthsPassed;
		bool nodeCountPassed;

		virtual void print(std::ostream& s) const override;
		void runTests() override;
		virtual void write(const fs::path& outDir) const override;
	};

	using IBuildResPtr = std::shared_ptr<IBuildRes>;

	template<typename Coord>
	class InterBuildRes : public IBuildRes {
	protected:
		const AlgoBuildRes& getRefRes() const override;
		const AlgoBuildRes& getSubRes() const override;

	public:
		InterErrPtr<Coord> err;
		AlgoBuildRes refRes;
		AlgoBuildRes subRes;

		InterBuildRes(const size_t nodeCount);
		void print(std::ostream& s) const override;
		void write(const fs::path& outDir) const override;
	};

	template<typename Coord>
	using InterBuildResPtr = std::shared_ptr<InterBuildRes<Coord>>;

	void printTestRes(const std::string& title, const bool passed, std::ostream& s);

	struct SeqAlgoBuildRes : public AlgoBuildRes {
		chr::microseconds total;

		void print(std::ostream& s, const std::string& title) const override;
		SeqAlgoBuildRes(const size_t nodeCount);
	};

	class SeqBuildRes : public IBuildRes {
	protected:
		const AlgoBuildRes& getRefRes() const override;
		const AlgoBuildRes& getSubRes() const override;

	public:
		SeqAlgoBuildRes refRes;
		SeqAlgoBuildRes subRes;

		SeqBuildRes(const size_t nodeCount);
	};

	using SeqBuildResPtr = std::shared_ptr<SeqBuildRes>;

	void writeConn(const IdxVec3DPtr& conn, std::ostream& stream);
	void writeConn(const IdxVec3DPtr& conn, const fs::path& path);

	template<typename Coord>
	inline const AlgoBuildRes& InterBuildRes<Coord>::getRefRes() const {
		return this->refRes;
	}

	template<typename Coord>
	inline const AlgoBuildRes& InterBuildRes<Coord>::getSubRes() const {
		return this->subRes;
	}

	template<typename Coord>
	inline InterBuildRes<Coord>::InterBuildRes(const size_t nodeCount) : IBuildRes(nodeCount), refRes(nodeCount), subRes(nodeCount) {}

	template<typename Coord>
	inline void InterBuildRes<Coord>::print(std::ostream& s) const {
		this->printTests(s);
		printTestRes("Intermediates comparison", !this->err, s);
		this->printTime(s);
	}

	template<typename Coord>
	inline void InterBuildRes<Coord>::write(const fs::path& outDir) const {
		IBuildRes::write(outDir);

		if(this->err)
			this->err->write(outDir);
	}
}
