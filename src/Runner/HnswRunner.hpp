#pragma once
#include "HnswRunCfg.hpp"
#include "ISearchRes.hpp"
#include "trueNeighbors.hpp"

namespace chm {
	template<typename Coord>
	class HnswRunner : public Unique {
	protected:
		const HnswCfgPtr algoCfg;
		const ICoordsPtr<Coord> coords;
		const HnswRunCfgPtr runCfg;

		size_t getNodeCount() const;
		HnswRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg);

	public:
		virtual IBuildResPtr build() = 0;
		virtual ISearchResPtr<Coord> search(const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors) = 0;
	};

	template<typename Coord>
	using HnswRunnerPtr = std::shared_ptr<HnswRunner<Coord>>;

	template<typename Coord>
	class HnswInterRunner : public HnswRunner<Coord> {
		size_t curIdx;
		InterErrPtr<Coord> err;
		IHnswIntermediatePtr<Coord> refAlgo;
		chr::microseconds refTime;
		IHnswIntermediatePtr<Coord> subAlgo;
		chr::microseconds subTime;

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
		IBuildResPtr build() override;
		InterBuildResPtr<Coord> buildInter();
		ISearchResPtr<Coord> search(const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors) override;
		InterSearchResPtr<Coord> searchInter(const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors);
	};

	template<typename Coord>
	class HnswSeqRunner : public HnswRunner<Coord> {
		IHnswPtr<Coord> refAlgo;
		IHnswPtr<Coord> subAlgo;

		IHnswPtr<Coord> build(const HnswTypePtr& type, SeqAlgoBuildRes& res);
		void search(
			IHnswPtr<Coord>& hnsw, const SearchCfgPtr<Coord>& searchCfg, SeqAlgoSearchRes<Coord>& res, const FoundNeighborsPtr<Coord>& trueNeighbors
		);

	public:
		HnswSeqRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg);
		IBuildResPtr build() override;
		SeqBuildResPtr buildSeq();
		ISearchResPtr<Coord> search(const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors) override;
		SeqSearchResPtr<Coord> searchSeq(const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors);
	};

	template<typename Coord>
	HnswRunnerPtr<Coord> createRunner(
		const HnswCfgPtr& algoCfg, const bool checkIntermediates, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg
	);

	template<typename Coord>
	inline size_t HnswRunner<Coord>::getNodeCount() const {
		return this->coords->get()->size() / this->algoCfg->dim;
	}

	template<typename Coord>
	inline HnswRunner<Coord>::HnswRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg)
		: algoCfg(algoCfg), coords(coords), runCfg(runCfg) {}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::insert(const ConstIter<Coord>& query) {
		this->startInsert(query);
		this->testLatestLevel();
		this->prepareUpperSearch();

		{
			const auto rng = this->refAlgo->getUpperRange();

			if(rng)
				for(auto lc = rng->start; lc > rng->end; lc--) {
					this->searchUpperLayers(lc);
					this->testNearestNode(lc);
				}
		}

		const auto rng = this->refAlgo->getLowerRange();

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
		this->err = std::make_shared<InterErr<Coord>>(this->curIdx, expected, actual, lc, test);
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
		this->refAlgo->startInsert(query);
		this->refTime += timer.stop();

		timer.start();
		this->subAlgo->startInsert(query);
		this->subTime += timer.stop();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testLatestLevel() {
		if(this->err)
			return;

		const auto actual = this->subAlgo->getLatestLevel();
		const auto expected = this->refAlgo->getLatestLevel();

		if(actual != expected)
			this->setErr(actual, expected, InterTest::LEVEL);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::prepareUpperSearch() {
		Timer timer{};
		timer.start();
		this->refAlgo->prepareUpperSearch();
		this->refTime += timer.stop();

		timer.start();
		this->subAlgo->prepareUpperSearch();
		this->subTime += timer.stop();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::searchUpperLayers(const size_t lc) {
		Timer timer{};
		timer.start();
		this->refAlgo->searchUpperLayers(lc);
		this->refTime += timer.stop();

		timer.start();
		this->subAlgo->searchUpperLayers(lc);
		this->subTime += timer.stop();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testNearestNode(const size_t lc) {
		if(this->err)
			return;

		const auto actual = this->subAlgo->getNearestNode();
		const auto expected = this->refAlgo->getNearestNode();

		if(actual.dist != expected.dist || actual.idx != expected.idx)
			this->setErr(actual, expected, lc, InterTest::UPPER_SEARCH);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::prepareLowerSearch() {
		Timer timer{};
		timer.start();
		this->refAlgo->prepareLowerSearch();
		this->refTime += timer.stop();

		timer.start();
		this->subAlgo->prepareLowerSearch();
		this->subTime += timer.stop();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testLowerSearchEntry() {
		if(this->err)
			return;

		const auto actual = this->subAlgo->getLowerSearchEntry();
		const auto expected = this->refAlgo->getLowerSearchEntry();

		if(actual != expected)
			this->setErr(actual, expected, InterTest::LOWER_SEARCH_ENTRY);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::searchLowerLayers(const size_t lc) {
		Timer timer{};
		timer.start();
		this->refAlgo->searchLowerLayers(lc);
		this->refTime += timer.stop();

		timer.start();
		this->subAlgo->searchLowerLayers(lc);
		this->subTime += timer.stop();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testLowerLayerResults(const size_t lc) {
		if(this->err)
			return;

		this->testVec(this->subAlgo->getLowerLayerResults(), this->refAlgo->getLowerLayerResults(), lc, InterTest::LOWER_SEARCH_RES);
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::selectOriginalNeighbors(const size_t lc) {
		Timer timer{};
		timer.start();
		this->refAlgo->selectOriginalNeighbors(lc);
		this->refTime += timer.stop();

		timer.start();
		this->subAlgo->selectOriginalNeighbors(lc);
		this->subTime += timer.stop();
	}

	template<typename Coord>
	inline BigNodeVecPtr<Coord> HnswInterRunner<Coord>::testOriginalNeighbors(const size_t lc) {
		if(this->err)
			return nullptr;

		const auto expected = this->refAlgo->getOriginalNeighbors();
		this->testVec(this->subAlgo->getOriginalNeighbors(), expected, lc, InterTest::SELECTED_NEIGHBORS);
		return expected;
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::connect(const size_t lc) {
		Timer timer{};
		timer.start();
		this->refAlgo->connect(lc);
		this->refTime += timer.stop();

		timer.start();
		this->subAlgo->connect(lc);
		this->subTime += timer.stop();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testConnections(const size_t nodeIdx, const size_t lc) {
		if(this->err)
			return;

		auto actual = std::make_shared<BigNodeVec<Coord>>();
		auto expected = std::make_shared<BigNodeVec<Coord>>();
		const auto refNeighbors = this->refAlgo->getNeighborsForNode(nodeIdx, lc);
		const auto subNeighbors = this->subAlgo->getNeighborsForNode(nodeIdx, lc);

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
		this->refAlgo->setupEnterPoint();
		this->refTime += timer.stop();

		timer.start();
		this->subAlgo->setupEnterPoint();
		this->subTime += timer.stop();
	}

	template<typename Coord>
	inline void HnswInterRunner<Coord>::testEnterPoint() {
		if(this->err)
			return;

		const auto actual = this->subAlgo->getEnterPoint();
		const auto expected = this->refAlgo->getEnterPoint();

		if(actual != expected)
			this->setErr(actual, expected, InterTest::ENTRY);
	}

	template<typename Coord>
	inline HnswInterRunner<Coord>::HnswInterRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg)
		: HnswRunner<Coord>(algoCfg, coords, runCfg), curIdx(0), err(nullptr), refAlgo(nullptr), refTime(0), subAlgo(nullptr), subTime(0) {}

	template<typename Coord>
	inline IBuildResPtr HnswInterRunner<Coord>::build() {
		return this->buildInter();
	}

	template<typename Coord>
	inline InterBuildResPtr<Coord> HnswInterRunner<Coord>::buildInter() {
		const auto coordsPtr = this->coords->get();
		const auto& coords = *coordsPtr;
		const auto len = this->getNodeCount();
		auto res = std::make_shared<InterBuildRes<Coord>>(len);
		Timer timer{};

		timer.start();
		this->refAlgo = createHnswIntermediate<Coord>(this->algoCfg, this->runCfg->refType);
		res->refRes.init = timer.stop();

		timer.start();
		this->subAlgo = createHnswIntermediate<Coord>(this->algoCfg, this->runCfg->subType);
		res->subRes.init = timer.stop();

		ProgressBar bar("Inserting elements.", len);

		for(size_t i = 0; i < len; i++) {
			this->curIdx = i;
			this->refTime = chr::microseconds(0);
			this->subTime = chr::microseconds(0);
			this->insert(coords.cbegin() + i * this->algoCfg->dim);

			res->refRes.queryTime.queries.push_back(this->refTime);
			res->subRes.queryTime.queries.push_back(this->subTime);
			bar.update();
		}

		res->err = err;
		res->refRes.queryTime.calcStats();
		res->refRes.conn = sortedInPlace(this->refAlgo->getConnections());
		res->subRes.queryTime.calcStats();
		res->subRes.conn = sortedInPlace(this->subAlgo->getConnections());
		res->runTests();
		return res;
	}

	template<typename Coord>
	inline ISearchResPtr<Coord> HnswInterRunner<Coord>::search(const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors) {
		return this->searchInter(cfg, trueNeighbors);
	}

	template<typename Coord>
	inline InterSearchResPtr<Coord> HnswInterRunner<Coord>::searchInter(
		const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors
	) {
		throw std::runtime_error("Method HnswInterRunner::searchInter isn't implemented.");
	}

	template<typename Coord>
	inline IHnswPtr<Coord> HnswSeqRunner<Coord>::build(const HnswTypePtr& type, SeqAlgoBuildRes& res) {
		const auto coordsPtr = this->coords->get();
		const auto& coords = *coordsPtr;
		const auto len = this->getNodeCount();
		ProgressBar bar("Inserting elements.", len);
		Timer timer{};
		Timer totalTimer{};
		totalTimer.start();

		timer.start();
		auto hnsw = createHnsw<Coord>(this->algoCfg, type);
		res.init = timer.stop();

		for(size_t i = 0; i < len; i++) {
			timer.start();
			hnsw->insert(coords.cbegin() + i * this->algoCfg->dim);
			const auto insTime = timer.stop();

			res.queryTime.queries.push_back(insTime);
			bar.update();
		}

		res.total = totalTimer.stop();
		res.queryTime.calcStats();
		return hnsw;
	}

	template<typename Coord>
	inline void HnswSeqRunner<Coord>::search(
		IHnswPtr<Coord>& hnsw, const SearchCfgPtr<Coord>& searchCfg, SeqAlgoSearchRes<Coord>& res, const FoundNeighborsPtr<Coord>& trueNeighbors
	) {
		const auto queryCoords = searchCfg->coords->get();
		const auto queryCount = queryCoords->size() / this->algoCfg->dim;
		ProgressBar bar("Searching queries.", queryCount);
		Timer timer{};
		Timer totalTimer{};
		totalTimer.start();

		for(size_t i = 0; i < queryCount; i++) {
			timer.start();
			hnsw->knnSearch(
				queryCoords->cbegin() + i * this->algoCfg->dim, searchCfg->K, searchCfg->ef, res.neighbors.indices[i], res.neighbors.distances[i]
			);
			const auto searchTime = timer.stop();

			res.queryTime.queries.push_back(searchTime);
			bar.update();
		}

		res.total = totalTimer.stop();
		res.calcRecall(trueNeighbors->indices);
		res.queryTime.calcStats();
	}

	template<typename Coord>
	inline HnswSeqRunner<Coord>::HnswSeqRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg)
		: HnswRunner<Coord>(algoCfg, coords, runCfg) {}

	template<typename Coord>
	inline IBuildResPtr HnswSeqRunner<Coord>::build() {
		return this->buildSeq();
	}

	template<typename Coord>
	inline SeqBuildResPtr HnswSeqRunner<Coord>::buildSeq() {
		auto res = std::make_shared<SeqBuildRes>(this->getNodeCount());
		this->refAlgo = this->build(this->runCfg->refType, res->refRes);
		this->subAlgo = this->build(this->runCfg->subType, res->subRes);
		res->refRes.conn = sortedInPlace(this->refAlgo->getConnections());
		res->subRes.conn = sortedInPlace(this->subAlgo->getConnections());
		res->runTests();
		return res;
	}

	template<typename Coord>
	inline ISearchResPtr<Coord> HnswSeqRunner<Coord>::search(const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors) {
		return this->searchSeq(cfg, trueNeighbors);
	}

	template<typename Coord>
	inline SeqSearchResPtr<Coord> HnswSeqRunner<Coord>::searchSeq(const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors) {
		auto res = std::make_shared<SeqSearchRes<Coord>>(cfg->coords->get()->size() / this->algoCfg->dim);
		this->search(this->refAlgo, cfg, res->refRes, trueNeighbors);
		this->search(this->subAlgo, cfg, res->subRes, trueNeighbors);
		res->runTests();
		return res;
	}

	template<typename Coord>
	HnswRunnerPtr<Coord> createRunner(
		const HnswCfgPtr& algoCfg, const bool checkIntermediates, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg
	) {
		if(checkIntermediates)
			return std::make_shared<HnswInterRunner<Coord>>(algoCfg, coords, runCfg);
		return std::make_shared<HnswSeqRunner<Coord>>(algoCfg, coords, runCfg);
	}
}
