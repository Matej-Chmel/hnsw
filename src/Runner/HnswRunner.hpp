#pragma once
#include <filesystem>
#include <fstream>
#include <ios>
#include <numeric>
#include "HnswIntermediate.hpp"
#include "ProgressBar.hpp"

namespace chm {
	namespace fs = std::filesystem;

	template<typename Coord>
	class ICoords : public Unique {
		VecPtr<Coord> coords;

		virtual VecPtr<Coord> create() const = 0;

	public:
		virtual ~ICoords() = default;
		ICoords();
		VecPtr<Coord> get();
	};

	template<typename Coord>
	using ICoordsPtr = std::shared_ptr<ICoords<Coord>>;

	template<typename Coord>
	class RndCoords : public ICoords<Coord> {
		size_t count;
		size_t dim;
		Coord min;
		Coord max;
		unsigned int seed;

		VecPtr<Coord> create() const override;

	public:
		RndCoords(const size_t count, const size_t dim, const Coord min, const Coord max, const unsigned int seed);
	};

	template<typename Coord>
	class ReadCoords : public ICoords<Coord> {
		fs::path p;

		VecPtr<Coord> create() const override;

	public:
		ReadCoords(const fs::path& p);
	};

	using LL = long long;

	struct BuildTimeRes : public Unique {
		double avgInsertMS;
		LL initMS;
		std::vector<LL> insertMS;
		LL totalMS;

		BuildTimeRes(const size_t nodeCount);
		void calcStats();
		void print(const std::string& name, std::ostream& s) const;
	};

	enum class InterTest {
		ENTRY,
		FINAL_NEIGHBORS,
		LEVEL,
		LOWER_SEARCH_ENTRY,
		LOWER_SEARCH_RES,
		SELECTED_NEIGHBORS,
		UPPER_SEARCH
	};

	const char* const str(const InterTest t);

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
	class RunRes : public Unique {
		bool testLevels() const;
		bool testNeighborLengths() const;
		bool testNeighborIndices() const;
		bool testNodeCount(const size_t nodeCount) const;

	public:
		InterErrPtr<Coord> interErr;
		bool levelsPassed;
		bool neighborIndicesPassed;
		bool neighborLengthsPassed;
		bool nodeCountPassed;
		BuildTimeRes refBuild;
		IdxVec3DPtr refConn;
		BuildTimeRes subBuild;
		IdxVec3DPtr subConn;

		void print(std::ostream& s) const;
		RunRes(const size_t nodeCount);
		void runTests(const size_t nodeCount);
		void write(const fs::path& outDir) const;
	};

	template<typename Coord>
	using RunResPtr = std::shared_ptr<RunRes<Coord>>;

	enum class HnswKind {
		CHM_AUTO,
		CHM_INT,
		CHM_SIZE_T,
		CHM_SHORT,
		HNSWLIB
	};

	struct HnswType : public Unique {
		const bool isIntermediate;
		const HnswKind kind;

		HnswType(const bool isIntermediate, const HnswKind kind);
	};

	using HnswTypePtr = std::shared_ptr<HnswType>;

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg);

	template<typename Coord, bool useEuclid>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg);

	template<typename Coord, typename Idx>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg);

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg, const HnswTypePtr& type);

	template<typename Coord, typename Idx>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg);

	template<typename Coord>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg, const HnswTypePtr& type);

	struct HnswRunCfg : public Unique {
		const HnswTypePtr refType;
		const HnswTypePtr subType;

		HnswRunCfg(const HnswTypePtr& refType, const HnswTypePtr& subType);
	};

	using HnswRunCfgPtr = std::shared_ptr<HnswRunCfg>;

	template<typename Coord>
	class HnswRunner : public Unique {
		IdxVec3DPtr build(const HnswTypePtr& type, BuildTimeRes& res);

	protected:
		const HnswCfgPtr algoCfg;
		const ICoordsPtr<Coord> coords;
		const HnswRunCfgPtr runCfg;

		size_t getNodeCount() const;

	public:
		HnswRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg);
		virtual RunResPtr<Coord> run();
	};

	template<typename Coord>
	using HnswRunPtr = std::shared_ptr<HnswRunner<Coord>>;

	template<typename Coord>
	class HnswInterRunner : public HnswRunner<Coord> {
		size_t curIdx;
		InterErrPtr<Coord> err;
		IHnswIntermediatePtr<Coord> ref;
		LL refTime;
		IHnswIntermediatePtr<Coord> sub;
		LL subTime;

		void insert(const ConstIter<Coord>& query);
		void setErr(const size_t actual, const size_t expected, const InterTest test);
		void setErr(const BigNode<Coord>& actual, const BigNode<Coord>& expected, const size_t lc, const InterTest test);
		void setErr(const BigNodeVecPtr<Coord>& actual, const BigNodeVecPtr<Coord>& expected, const size_t lc, const InterTest test);
		void testVec(BigNodeVecPtr<Coord> actual, BigNodeVecPtr<Coord> expected, const size_t lc, const InterTest test);

		void startInsert(const ConstIter<Coord>& query);
		void testLatestLevel();
		void prepareUpperSearch();
		void searchUpperLayers(const size_t lc);
		void testNearestNode(const size_t lc);
		void prepareLowerSearch();
		void testLowerSearchEntry();
		void searchLowerLayers(const size_t lc);
		void testLowerLayerResults(const size_t lc);
		void selectOriginalNeighbors(const size_t lc);
		BigNodeVecPtr<Coord> testOriginalNeighbors(const size_t lc);
		void connect(const size_t lc);
		void testConnections(const size_t nodeIdx, const size_t lc);
		void testConnections(const BigNodeVecPtr<Coord>& neighbors, const size_t queryIdx, const size_t lc);
		void setupEnterPoint();
		void testEnterPoint();

	public:
		HnswInterRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg);
		RunResPtr<Coord> run() override;
	};

	template<typename Coord>
	HnswRunPtr<Coord> createRunner(
		const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const bool isIntermediate, const HnswRunCfgPtr& runCfg
	);

	void printTestRes(const std::string& title, const bool passed, std::ostream& s);
	IdxVec3DPtr sortedInPlace(const IdxVec3DPtr& conn);
	void writeConn(const IdxVec3DPtr& conn, std::ostream& stream);
	void writeConn(const IdxVec3DPtr& conn, const fs::path& path);

	template<typename Coord>
	void writeVec(const BigNodeVecPtr<Coord>& vec, std::ostream& stream);

	template<typename Coord>
	void writeVec(const BigNodeVecPtr<Coord>& vec, const fs::path& path);

	class Timer : public Unique {
		chr::system_clock::time_point from;

	public:
		void start();
		LL stopMS();
		Timer();
	};

	template<typename Coord>
	using UniformDistribution = std::conditional_t<
		std::is_integral<Coord>::value, std::uniform_int_distribution<Coord>,
		std::conditional_t<std::is_floating_point<Coord>::value, std::uniform_real_distribution<Coord>, void>
	>;

	template<typename Coord>
	inline ICoords<Coord>::ICoords() : coords(nullptr) {}

	template<typename Coord>
	inline VecPtr<Coord> ICoords<Coord>::get() {
		if(!this->coords)
			this->coords = this->create();
		return this->coords;
	}

	template<typename Coord>
	inline VecPtr<Coord> RndCoords<Coord>::create() const {
		UniformDistribution<Coord> dist(this->min, this->max);
		std::default_random_engine gen(this->seed);

		auto res = std::make_shared<std::vector<Coord>>();
		auto& r = *res;
		const auto total = this->count * this->dim;

		ProgressBar bar("Generating coordinates.", total, 32);
		r.reserve(total);

		for(size_t i = 0; i < total; i++) {
			r.push_back(dist(gen));
			bar.update();
		}

		return res;
	}

	template<typename Coord>
	inline RndCoords<Coord>::RndCoords(const size_t count, const size_t dim, const Coord min, const Coord max, const unsigned int seed)
		: count(count), dim(dim), min(min), max(max), seed(seed) {}

	template<typename Coord>
	inline VecPtr<Coord> ReadCoords<Coord>::create() const {
		std::ifstream s(this->p);

		s.seekg(0, std::ios::end);
		const auto fileSize = s.tellg();
		s.seekg(0, std::ios::beg);

		auto res = std::make_shared<std::vector<Coord>>();
		res->resize(fileSize / sizeof(Coord));
		s.read(reinterpret_cast<std::ifstream::char_type*>(res->data()), fileSize);
		return res;
	}

	template<typename Coord>
	inline ReadCoords<Coord>::ReadCoords(const fs::path& p) : p(p) {}

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg) {
		if(cfg->useEuclid)
			return createHnswIntermediate<Coord, true>(cfg);
		return createHnswIntermediate<Coord, false>(cfg);
	}

	template<typename Coord, bool useEuclid>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg) {
		if(isWideEnough<unsigned short>(cfg->maxNodeCount))
			return std::make_shared<HnswInterImpl<Coord, unsigned short, useEuclid>>(cfg);
		if(isWideEnough<unsigned int>(cfg->maxNodeCount))
			return std::make_shared<HnswInterImpl<Coord, unsigned int, useEuclid>>(cfg);
		return std::make_shared<HnswInterImpl<Coord, size_t, useEuclid>>(cfg);
	}

	template<typename Coord, typename Idx>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg) {
		if(cfg->useEuclid)
			return std::make_shared<HnswInterImpl<Coord, Idx, true>>(cfg);
		return std::make_shared<HnswInterImpl<Coord, Idx, false>>(cfg);
	}

	template<typename Coord>
	IHnswIntermediatePtr<Coord> createHnswIntermediate(const HnswCfgPtr& cfg, const HnswTypePtr& type) {
		switch(type->kind) {
			case HnswKind::CHM_AUTO:
				return createHnswIntermediate<Coord>(cfg);
			case HnswKind::CHM_INT:
				return createHnswIntermediate<Coord, unsigned int>(cfg);
			case HnswKind::CHM_SHORT:
				return createHnswIntermediate<Coord, unsigned short>(cfg);
			case HnswKind::CHM_SIZE_T:
				return createHnswIntermediate<Coord, size_t>(cfg);
			case HnswKind::HNSWLIB:
				return std::make_shared<HnswlibInterImpl<Coord>>(cfg);
			default:
				return nullptr;
		}
	}

	template<typename Coord, typename Idx>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg) {
		if(cfg->useEuclid)
			return std::make_shared<Hnsw<Coord, Idx, true>>(cfg);
		return std::make_shared<Hnsw<Coord, Idx, false>>(cfg);
	}

	template<typename Coord>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg, const HnswTypePtr& type) {
		if(type->isIntermediate)
			return createHnswIntermediate<Coord>(cfg, type);

		switch(type->kind) {
			case HnswKind::CHM_AUTO:
				return createHnsw<Coord>(cfg);
			case HnswKind::CHM_INT:
				return createHnsw<Coord, unsigned int>(cfg);
			case HnswKind::CHM_SHORT:
				return createHnsw<Coord, unsigned short>(cfg);
			case HnswKind::CHM_SIZE_T:
				return createHnsw<Coord, size_t>(cfg);
			case HnswKind::HNSWLIB:
				return std::make_shared<HnswlibWrapper<Coord>>(cfg);
			default:
				return nullptr;
		}
	}

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
	inline void RunRes<Coord>::print(std::ostream& s) const {
		printTestRes("Node count", this->nodeCountPassed, s);
		printTestRes("Levels", this->levelsPassed, s);
		printTestRes("Neighbors lengths", this->neighborLengthsPassed, s);
		printTestRes("Neighbors indices", this->neighborIndicesPassed, s);
		printTestRes("Intermediate", !this->interErr, s);
		this->refBuild.print("Reference algorithm", s);
		this->subBuild.print("Subject algorithm", s);
	}

	template<typename Coord>
	inline bool RunRes<Coord>::testLevels() const {
		const auto len = std::min(this->refConn->size(), this->subConn->size());

		for(size_t i = 0; i < len; i++)
			if((*this->refConn)[i].size() != (*this->subConn)[i].size())
				return false;

		return true;
	}

	template<typename Coord>
	inline bool RunRes<Coord>::testNeighborLengths() const {
		const auto len = std::min(this->refConn->size(), this->subConn->size());

		for(size_t i = 0; i < len; i++) {
			const auto& nodeLayersRef = (*this->refConn)[i];
			const auto& nodeLayersSub = (*this->subConn)[i];
			const auto nodeLayersLen = std::min(nodeLayersRef.size(), nodeLayersSub.size());

			for(size_t layerIdx = 0; layerIdx < nodeLayersLen; layerIdx++)
				if(nodeLayersRef[layerIdx].size() != nodeLayersSub[layerIdx].size())
					return false;
		}

		return true;
	}

	template<typename Coord>
	inline bool RunRes<Coord>::testNeighborIndices() const {
		const auto len = std::min(this->refConn->size(), this->subConn->size());

		for(size_t i = 0; i < len; i++) {
			const auto& nodeLayersRef = (*this->refConn)[i];
			const auto& nodeLayersSub = (*this->subConn)[i];
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

	template<typename Coord>
	inline bool RunRes<Coord>::testNodeCount(const size_t nodeCount) const {
		return this->refConn->size() == nodeCount && this->subConn->size() == nodeCount;
	}

	template<typename Coord>
	inline RunRes<Coord>::RunRes(const size_t nodeCount)
		: interErr(nullptr), levelsPassed(false), neighborIndicesPassed(false), neighborLengthsPassed(false), nodeCountPassed(false),
		refBuild(nodeCount), refConn(nullptr), subBuild(nodeCount), subConn(nullptr) {
	}

	template<typename Coord>
	inline void RunRes<Coord>::runTests(const size_t nodeCount) {
		this->levelsPassed = this->testLevels();
		this->neighborIndicesPassed = this->testNeighborIndices();
		this->neighborLengthsPassed = this->testNeighborLengths();
		this->nodeCountPassed = this->testNodeCount(nodeCount);
	}

	template<typename Coord>
	inline void RunRes<Coord>::write(const fs::path& outDir) const {
		for(const auto& entry : fs::directory_iterator(outDir))
			if(entry.is_regular_file())
				fs::remove(entry.path());

		writeConn(this->refConn, outDir / "refConn.log");
		writeConn(this->subConn, outDir / "subConn.log");

		if(this->interErr)
			this->interErr->write(outDir);

		std::ofstream s(outDir / "run.log");
		this->print(s);
	}

	template<typename Coord>
	inline IdxVec3DPtr HnswRunner<Coord>::build(const HnswTypePtr& type, BuildTimeRes& res) {
		Timer timer{};

		timer.start();
		auto hnsw = createHnsw<Coord>(this->algoCfg, type);
		res.initMS = timer.stopMS();

		const auto coordsPtr = this->coords->get();
		const auto& coords = *coordsPtr;
		const auto len = this->getNodeCount();

		ProgressBar bar("Inserting elements.", len, 32);

		for(size_t i = 0; i < len; i++) {
			timer.start();
			hnsw->insert(coords.cbegin() + i * this->algoCfg->dim);
			const auto insTime = timer.stopMS();

			res.insertMS.push_back(insTime);
			bar.update();
		}

		res.calcStats();
		return sortedInPlace(hnsw->getConnections());
	}

	template<typename Coord>
	inline size_t HnswRunner<Coord>::getNodeCount() const {
		return this->coords->get()->size() / this->algoCfg->dim;
	}

	template<typename Coord>
	inline HnswRunner<Coord>::HnswRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg)
		: algoCfg(algoCfg), coords(coords), runCfg(runCfg) {}

	template<typename Coord>
	inline RunResPtr<Coord> HnswRunner<Coord>::run() {
		auto res = std::make_shared<RunRes<Coord>>(this->getNodeCount());
		res->refConn = this->build(this->runCfg->refType, res->refBuild);
		res->subConn = this->build(this->runCfg->subType, res->subBuild);
		res->runTests(this->getNodeCount());
		return res;
	}

	template<typename Coord>
	HnswRunPtr<Coord> createRunner(
		const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const bool isIntermediate, const HnswRunCfgPtr& runCfg
	) {
		if(isIntermediate)
			return std::make_shared<HnswInterRunner<Coord>>(algoCfg, coords, runCfg);
		return std::make_shared<HnswRunner<Coord>>(algoCfg, coords, runCfg);
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

	template<typename Coord>
	inline void HnswInterRunner<Coord>::insert(const ConstIter<Coord>& query) {
		this->startInsert(query);
		this->testLatestLevel();
		this->prepareUpperSearch();

		{
			const auto rng = this->ref->getUpperRange();

			if(rng)
				for(auto lc = rng->start; lc > rng->end; lc--) {
					this->searchUpperLayers(lc);
					this->testNearestNode(lc);
				}
		}

		const auto rng = this->ref->getLowerRange();

		if(rng) {
			this->prepareLowerSearch();

			for(auto lc = rng->start;; lc--) {
				this->testLowerSearchEntry();
				this->searchLowerLayers(lc);
				this->testLowerLayerResults(lc);

				this->selectOriginalNeighbors(lc);
				const auto neighbors = this->testOriginalNeighbors(lc);

				this->connect(lc);
				this->testConnections(neighbors, this->curIdx, lc);

				if(!lc)
					break;
			}
		}

		this->setupEnterPoint();
		this->testEnterPoint();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::setErr(const size_t actual, const size_t expected, const InterTest test) {
		this->setErr(BigNode<Coord>(0, actual), BigNode<Coord>(0, expected), 0, test);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::setErr(const BigNode<Coord>& actual, const BigNode<Coord>& expected, const size_t lc, const InterTest test) {
		this->setErr(
			std::make_shared<BigNodeVec<Coord>>(BigNodeVec<Coord>{actual}),
			std::make_shared<BigNodeVec<Coord>>(BigNodeVec<Coord>{expected}),
			lc, test
		);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::setErr(
		const BigNodeVecPtr<Coord>& actual, const BigNodeVecPtr<Coord>& expected, const size_t lc, const InterTest test
	) {
		this->err = std::make_shared<InterErr<Coord>>(this->curIdx, actual, expected, lc, test);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testVec(
		BigNodeVecPtr<Coord> actual, BigNodeVecPtr<Coord> expected, const size_t lc, const InterTest test
	) {
		auto& a = *actual;
		auto& e = *expected;
		const auto len = e.size();

		std::sort(a.begin(), a.end(), BigNodeCmp<Coord>());
		std::sort(e.begin(), e.end(), BigNodeCmp<Coord>());

		if(a.size() != len) {
			this->setErr(actual, expected, lc, test);
			return;
		}

		for(size_t i = 0; i < len; i++) {
			const auto& aNode = a[i];
			const auto& eNode = e[i];

			if(aNode.dist != eNode.dist || aNode.idx != eNode.idx) {
				this->setErr(actual, expected, lc, test);
				return;
			}
		}
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::startInsert(const ConstIter<Coord>& query) {
		Timer timer{};
		timer.start();
		this->ref->startInsert(query);
		this->refTime += timer.stopMS();

		timer.start();
		this->sub->startInsert(query);
		this->subTime += timer.stopMS();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testLatestLevel() {
		if(this->err)
			return;

		const auto actual = this->sub->getLatestLevel();
		const auto expected = this->ref->getLatestLevel();

		if(actual != expected)
			this->setErr(actual, expected, InterTest::LEVEL);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::prepareUpperSearch() {
		Timer timer{};
		timer.start();
		this->ref->prepareUpperSearch();
		this->refTime += timer.stopMS();

		timer.start();
		this->sub->prepareUpperSearch();
		this->subTime += timer.stopMS();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::searchUpperLayers(const size_t lc) {
		Timer timer{};
		timer.start();
		this->ref->searchUpperLayers(lc);
		this->refTime += timer.stopMS();

		timer.start();
		this->sub->searchUpperLayers(lc);
		this->subTime += timer.stopMS();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testNearestNode(const size_t lc) {
		if(this->err)
			return;

		const auto actual = this->sub->getNearestNode();
		const auto expected = this->ref->getNearestNode();

		if(actual.dist != expected.dist || actual.idx != expected.idx)
			this->setErr(actual, expected, lc, InterTest::UPPER_SEARCH);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::prepareLowerSearch() {
		Timer timer{};
		timer.start();
		this->ref->prepareLowerSearch();
		this->refTime += timer.stopMS();

		timer.start();
		this->sub->prepareLowerSearch();
		this->subTime += timer.stopMS();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testLowerSearchEntry() {
		if(this->err)
			return;

		const auto actual = this->sub->getLowerSearchEntry();
		const auto expected = this->ref->getLowerSearchEntry();

		if(actual != expected)
			this->setErr(actual, expected, InterTest::LOWER_SEARCH_ENTRY);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::searchLowerLayers(const size_t lc) {
		Timer timer{};
		timer.start();
		this->ref->searchLowerLayers(lc);
		this->refTime += timer.stopMS();

		timer.start();
		this->sub->searchLowerLayers(lc);
		this->subTime += timer.stopMS();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testLowerLayerResults(const size_t lc) {
		if(this->err)
			return;

		this->testVec(this->sub->getLowerLayerResults(), this->ref->getLowerLayerResults(), lc, InterTest::LOWER_SEARCH_RES);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::selectOriginalNeighbors(const size_t lc) {
		Timer timer{};
		timer.start();
		this->ref->selectOriginalNeighbors(lc);
		this->refTime += timer.stopMS();

		timer.start();
		this->sub->selectOriginalNeighbors(lc);
		this->subTime += timer.stopMS();
	}

	template<typename Coord>
	inline BigNodeVecPtr<Coord> HnswInterRunner<Coord>::testOriginalNeighbors(const size_t lc) {
		if(this->err)
			return nullptr;

		const auto expected = this->ref->getOriginalNeighbors();
		this->testVec(this->sub->getOriginalNeighbors(), expected, lc, InterTest::SELECTED_NEIGHBORS);
		return expected;
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::connect(const size_t lc) {
		Timer timer{};
		timer.start();
		this->ref->connect(lc);
		this->refTime += timer.stopMS();

		timer.start();
		this->sub->connect(lc);
		this->subTime += timer.stopMS();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testConnections(const size_t nodeIdx, const size_t lc) {
		if(this->err)
			return;

		auto actual = std::make_shared<BigNodeVec<Coord>>();
		auto expected = std::make_shared<BigNodeVec<Coord>>();
		const auto refNeighbors = this->ref->getNeighborsForNode(nodeIdx, lc);
		const auto subNeighbors = this->sub->getNeighborsForNode(nodeIdx, lc);

		actual->reserve(subNeighbors->size());
		expected->reserve(refNeighbors->size());

		for(const auto& i : *subNeighbors)
			actual->emplace_back(0, i);
		for(const auto& i : *refNeighbors)
			expected->emplace_back(0, i);

		this->testVec(actual, expected, lc, InterTest::FINAL_NEIGHBORS);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testConnections(const BigNodeVecPtr<Coord>& neighbors, const size_t queryIdx, const size_t lc) {
		if(this->err)
			return;

		this->testConnections(queryIdx, lc);

		for(const auto& n : *neighbors)
			this->testConnections(n.idx, lc);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::setupEnterPoint() {
		Timer timer{};
		timer.start();
		this->ref->setupEnterPoint();
		this->refTime += timer.stopMS();

		timer.start();
		this->sub->setupEnterPoint();
		this->subTime += timer.stopMS();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testEnterPoint() {
		if(this->err)
			return;

		const auto actual = this->sub->getEnterPoint();
		const auto expected = this->ref->getEnterPoint();

		if(actual != expected)
			this->setErr(actual, expected, InterTest::ENTRY);
	}

	template<typename Coord>
	inline HnswInterRunner<Coord>::HnswInterRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg)
		: HnswRunner<Coord>(algoCfg, coords, runCfg), curIdx(0), err(nullptr), ref(nullptr), refTime(0), sub(nullptr), subTime(0) {}

	template<typename Coord>
	inline RunResPtr<Coord> HnswInterRunner<Coord>::run() {
		const auto coordsPtr = this->coords->get();
		const auto& coords = *coordsPtr;
		const auto len = this->getNodeCount();
		auto res = std::make_shared<RunRes<Coord>>(len);
		Timer timer{};

		timer.start();
		this->ref = createHnswIntermediate<Coord>(this->algoCfg, this->runCfg->refType);
		res->refBuild.initMS = timer.stopMS();

		timer.start();
		this->sub = createHnswIntermediate<Coord>(this->algoCfg, this->runCfg->subType);
		res->subBuild.initMS = timer.stopMS();

		ProgressBar bar("Inserting elements.", len, 32);

		for(size_t i = 0; i < len; i++) {
			this->curIdx = i;
			this->refTime = 0;
			this->subTime = 0;
			this->insert(coords.cbegin() + i * this->algoCfg->dim);

			res->refBuild.insertMS.push_back(this->refTime);
			res->subBuild.insertMS.push_back(this->subTime);
			bar.update();
		}

		res->refBuild.calcStats();
		res->refConn = sortedInPlace(this->ref->getConnections());
		res->subBuild.calcStats();
		res->subConn = sortedInPlace(this->sub->getConnections());
		res->runTests(len);
		return res;
	}
}
