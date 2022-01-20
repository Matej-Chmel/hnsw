#include <algorithm>
#include <cmath>
#include <unordered_set>
#include "Graph.hpp"

namespace chm {
	NodeVecPtr sortedNearest(FarHeap& h) {
		auto res = std::make_shared<NodeVec>(h.begin(), h.end());
		std::sort(res->begin(), res->end(), NodeComparator());
		return res;
	}

	Config::Config(const HNSWConfigPtr& cfg)
		: dim(cfg->dim), efConstruction(cfg->efConstruction), M(cfg->M), mL(1.0 / std::log(1.0 * this->M)), Mmax(cfg->M), Mmax0(cfg->M * 2) {}

	constexpr bool FarComparator::operator()(const Node& a, const Node& b) const noexcept {
		return a.dist < b.dist;
	}

	constexpr bool NearComparator::operator()(const Node& a, const Node& b) const noexcept {
		return a.dist > b.dist;
	}

	void Graph::connect(FarHeap& neighborHeap, IdxVec& resNeighbors) {
		const auto len = neighborHeap.len();
		resNeighbors.clear();
		resNeighbors.reserve(len);

		if(len < 2)
			resNeighbors.emplace_back(neighborHeap.top().idx);
		else {
			const auto shortLen = len - 2;

			for(size_t i = 0; i < shortLen; i++) {
				resNeighbors.emplace_back(neighborHeap.top().idx);
				neighborHeap.pop();
			}

			resNeighbors.emplace_back(neighborHeap.top().idx);
			resNeighbors.emplace_back(neighborHeap.back().idx);
		}
	}

	void Graph::fillHeap(const float* query, size_t newIdx, IdxVec& eConn, FarHeap& eNewConn) {
		eNewConn.clear();
		eNewConn.reserve(eConn.size());
		eNewConn.push(this->getDistance(this->getCoords(newIdx), query), newIdx);

		for(auto& idx : eConn)
			eNewConn.push(this->getDistance(this->getCoords(idx), query), idx);
	}

	const float* Graph::getCoords(size_t idx) {
		return &this->coords[idx * this->cfg.dim];
	}

	float Graph::getDistance(const float* node, const float* query, bool useCache, size_t nodeIdx) {
		if(useCache) {
			auto iter = this->distancesCache.find(nodeIdx);

			if(iter != this->distancesCache.end())
				return iter->second;
		}

		float res = 0.f;

		for(size_t i = 0; i < this->cfg.dim; i++) {
			float diff = node[i] - query[i];
			res += diff * diff;
		}

		if(useCache)
			this->distancesCache[nodeIdx] = res;

		return res;
	}

	IdxVec& Graph::getNeighbors(size_t idx, size_t lc) {
		return this->connections[idx][lc];
	}

	size_t Graph::getNewLevel() {
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		return size_t(
			-std::log(dist(this->gen)) * this->cfg.mL
		);
	}

	Graph::Graph(const Config& cfg, size_t maxNodeCount, unsigned int seed)
		: cfg(cfg), entryIdx(0), entryLevel(0), nodeCount(0) {

		this->coords.resize(cfg.dim * maxNodeCount);
		this->gen.seed(seed);
		this->connections.resize(maxNodeCount);
	}

	void Graph::initConnections(size_t queryIdx, size_t level) {
		size_t nLayers = level + 1;
		auto& qLayers = this->connections[queryIdx];
		qLayers.resize(nLayers);
	}

	void Graph::insert(const float* queryCoords, size_t queryIdx) {
		std::copy(queryCoords, queryCoords + this->cfg.dim, this->coords.begin() + this->nodeCount * this->cfg.dim);

		if(queryIdx == 0) {
			this->entryLevel = this->getNewLevel();
			this->nodeCount = 1;
			this->initConnections(this->entryIdx, this->entryLevel);
			return;
		}

		this->distancesCache.clear();
		this->nodeCount++;

		Node ep(this->getDistance(this->getCoords(this->entryIdx), queryCoords, true, this->entryIdx), this->entryIdx);
		auto L = this->entryLevel;
		auto l = this->getNewLevel();

		this->initConnections(queryIdx, l);

		for(auto lc = L; lc > l; lc--)
			this->searchUpperLayer(queryCoords, ep, lc);

		for(auto lc = std::min(L, l);; lc--) {
			FarHeap candidates;
			this->searchLowerLayer(queryCoords, ep, this->cfg.efConstruction, lc, candidates);
			this->selectNeighborsHeuristic(candidates, this->cfg.M);

			auto& neighbors = this->getNeighbors(queryIdx, lc);
			this->connect(candidates, neighbors);

			// ep = nearest from candidates
			ep = Node(candidates.back());
			auto layerMmax = !lc ? this->cfg.Mmax0 : this->cfg.Mmax;

			for(auto& eIdx : neighbors) {
				auto& eConn = this->getNeighbors(eIdx, lc);

				if(eConn.size() < layerMmax)
					eConn.push_back(queryIdx);
				else {
					this->fillHeap(this->getCoords(eIdx), queryIdx, eConn, candidates);
					this->selectNeighborsHeuristic(candidates, layerMmax);
					this->connect(candidates, eConn);
				}
			}

			if(!lc)
				break;
		}

		if(l > L) {
			this->entryIdx = queryIdx;
			this->entryLevel = l;
		}
	}

	void Graph::searchUpperLayer(const float* query, Node& resEp, size_t lc) {
		size_t prevIdx{};

		do {
			auto& neighbors = this->getNeighbors(resEp.idx, lc);
			prevIdx = resEp.idx;

			for(auto& cIdx : neighbors) {
				auto dist = this->getDistance(this->getCoords(cIdx), query, true, cIdx);

				if(dist < resEp.dist) {
					resEp.dist = dist;
					resEp.idx = cIdx;
				}
			}

		} while(resEp.idx != prevIdx);
	}

	void Graph::searchLowerLayer(const float* query, Node& ep, size_t ef, size_t lc, FarHeap& W) {
		NearHeap C{ep};
		std::unordered_set<size_t> v{ep.idx};
		W.push(ep);

		while(C.len()) {
			size_t cIdx;

			{
				auto& c = C.top();
				auto& f = W.top();

				if(c.dist > f.dist && W.len() == ef)
					break;

				cIdx = c.idx;
			}

			auto& neighbors = this->getNeighbors(cIdx, lc);

			// Extract nearest from C.
			C.pop();

			for(auto& eIdx : neighbors) {
				if(v.insert(eIdx).second) {
					auto eDistance = this->getDistance(this->getCoords(eIdx), query, true, eIdx);
					bool shouldAdd;

					{
						auto& f = W.top();
						shouldAdd = f.dist > eDistance || W.len() < ef;
					}

					if(shouldAdd) {
						C.push(eDistance, eIdx);
						W.push(eDistance, eIdx);

						if(W.len() > ef)
							W.pop();
					}
				}
			}
		}
	}

	void Graph::selectNeighborsHeuristic(FarHeap& outC, size_t M) {
		if(outC.len() < M)
			return;

		auto& R = outC;
		NearHeap W(outC);

		R.clear();
		R.reserve(std::min(W.len(), M));

		while(W.len() && R.len() < M) {
			{
				auto& e = W.top();
				auto eCoords = this->getCoords(e.idx);

				for(auto& rNode : R)
					if(this->getDistance(eCoords, this->getCoords(rNode.idx)) < e.dist)
						goto isNotCloser;

				R.push(e.dist, e.idx);
			}

			isNotCloser:;

			// Extract nearest from W.
			W.pop();
		}
	}

	void Graph::knnSearch(const float* query, size_t K, size_t ef, IdxVec& resIndices, FloatVec& resDistances) {
		this->distancesCache.clear();

		Node ep(this->getDistance(this->getCoords(this->entryIdx), query, true, this->entryIdx), this->entryIdx);
		auto L = this->entryLevel;

		for(size_t lc = L; lc > 0; lc--)
			this->searchUpperLayer(query, ep, lc);

		FarHeap W;
		this->searchLowerLayer(query, ep, ef, 0, W);

		while(W.len() > K)
			W.pop();

		const auto len = W.len();
		resDistances.resize(len);
		resIndices.resize(len);

		for(size_t i = len - 1;; i--) {
			{
				auto& n = W.top();
				resDistances[i] = n.dist;
				resIndices[i] = n.idx;
			}
			W.pop();

			if(!i)
				break;
		}
	}

	void GraphWrapper::insert(float* data, size_t idx) {
		this->hnsw->insert(data, idx);
	}

	GraphWrapper::~GraphWrapper() {
		delete this->hnsw;
	}

	IdxVec3DPtr GraphWrapper::getConnections() const {
		auto res = std::make_shared<IdxVec3D>(this->hnsw->connections.begin(), this->hnsw->connections.begin() + this->hnsw->nodeCount);

		for(auto& nodeLayers : *res)
			for(auto& layer : nodeLayers)
				std::sort(layer.begin(), layer.end());

		return res;
	}

	DebugHNSW* GraphWrapper::getDebugObject() {
		return nullptr;
	}

	GraphWrapper::GraphWrapper(const HNSWConfigPtr& cfg) : GraphWrapper(cfg, "chm-HNSW") {}

	GraphWrapper::GraphWrapper(const HNSWConfigPtr& cfg, const std::string& name) : HNSWAlgo(cfg, name), ef(0), hnsw(nullptr) {}

	void GraphWrapper::init() {
		this->hnsw = new Graph(Config(this->cfg), this->cfg->maxElements, this->cfg->seed);
	}

	KNNResultPtr GraphWrapper::search(const FloatVecPtr& coords, size_t K) {
		auto& c = *coords;
		auto count = coords->size() / this->cfg->dim;
		auto res = std::make_shared<KNNResult>();
		auto& r = *res;

		r.resize(count);

		for(size_t i = 0; i < count; i++) {
			auto& distances = r.distances[i];
			auto& indices = r.indices[i];
			this->hnsw->knnSearch(&c[i * this->cfg->dim], K, this->ef, indices, distances);
		}

		return res;
	}

	void GraphWrapper::setSearchEF(size_t ef) {
		this->ef = ef;
	}

	DebugGraph::DebugGraph(Graph* hnsw) : hnsw(hnsw), local{} {}

	void DebugGraph::startInsert(float* coords, size_t idx) {
		this->local.queryCoords = coords;
		this->local.queryIdx = idx;

		std::copy(
			this->local.queryCoords, this->local.queryCoords + this->hnsw->cfg.dim,
			this->hnsw->coords.begin() + this->hnsw->nodeCount * this->hnsw->cfg.dim
		);

		this->local.isFirstNode = !this->local.queryIdx;

		if(this->local.isFirstNode) {
			this->hnsw->entryLevel = this->hnsw->getNewLevel();
			this->hnsw->nodeCount = 1;
			this->hnsw->initConnections(this->hnsw->entryIdx, this->hnsw->entryLevel);
			return;
		}

		this->hnsw->distancesCache.clear();
		this->hnsw->nodeCount++;

		this->local.ep = {
			this->hnsw->getDistance(this->hnsw->getCoords(this->hnsw->entryIdx), this->local.queryCoords, true, this->hnsw->entryIdx),
			this->hnsw->entryIdx
		};
		this->local.L = this->hnsw->entryLevel;
		this->local.l = this->hnsw->getNewLevel();

		this->hnsw->initConnections(this->local.queryIdx, this->local.l);
	}

	size_t DebugGraph::getLatestLevel() {
		return this->local.l;
	}

	void DebugGraph::prepareUpperSearch() {
		// Nothing to prepare.
	}

	LevelRange DebugGraph::getUpperRange() {
		if(this->local.isFirstNode)
			return {0, 0, false};

		return {this->local.L, this->local.l, true};
	}

	void DebugGraph::searchUpperLayers(size_t lc) {
		this->hnsw->searchUpperLayer(this->local.queryCoords, this->local.ep, lc);
	}

	Node DebugGraph::getNearestNode() {
		return this->local.ep;
	}

	void DebugGraph::prepareLowerSearch() {
		// Nothing to prepare.
	}

	LevelRange DebugGraph::getLowerRange() {
		if(this->local.isFirstNode)
			return {0, 0, false};

		return {std::min(this->local.L, this->local.l), 0, true};
	}

	size_t DebugGraph::getLowerSearchEntry() {
		return this->local.ep.idx;
	}

	void DebugGraph::searchLowerLayers(size_t lc) {
		this->local.candidates = FarHeap{};
		this->hnsw->searchLowerLayer(this->local.queryCoords, this->local.ep, this->hnsw->cfg.efConstruction, lc, this->local.candidates);
	}

	NodeVecPtr DebugGraph::getLowerLayerResults() {
		return sortedNearest(this->local.candidates);
	}

	void DebugGraph::selectOriginalNeighbors(size_t lc) {
		this->hnsw->selectNeighborsHeuristic(this->local.candidates, this->hnsw->cfg.M);
	}

	NodeVecPtr DebugGraph::getOriginalNeighbors() {
		return sortedNearest(this->local.candidates);
	}

	void DebugGraph::connect(size_t lc) {
		auto& neighbors = this->hnsw->getNeighbors(this->local.queryIdx, lc);
		this->hnsw->connect(this->local.candidates, neighbors);

		// ep = nearest from candidates
		this->local.ep = Node(this->local.candidates.back());
		auto layerMmax = !lc ? this->hnsw->cfg.Mmax0 : this->hnsw->cfg.Mmax;

		for(auto& eIdx : neighbors) {
			auto& eConn = this->hnsw->getNeighbors(eIdx, lc);

			if(eConn.size() < layerMmax)
				eConn.push_back(this->local.queryIdx);
			else {
				this->hnsw->fillHeap(this->hnsw->getCoords(eIdx), this->local.queryIdx, eConn, this->local.candidates);
				this->hnsw->selectNeighborsHeuristic(this->local.candidates, layerMmax);
				this->hnsw->connect(this->local.candidates, eConn);
			}
		}
	}

	IdxVecPtr DebugGraph::getNeighborsForNode(size_t idx, size_t lc) {
		auto& neighbors = this->hnsw->getNeighbors(idx, lc);
		auto res = std::make_shared<IdxVec>(neighbors.begin(), neighbors.end());
		std::sort(res->begin(), res->end());
		return res;
	}

	void DebugGraph::prepareNextLayer(size_t lc) {
		// Nothing to prepare.
	}

	void DebugGraph::setupEnterPoint() {
		if(this->local.l > this->local.L) {
			this->hnsw->entryIdx = this->local.queryIdx;
			this->hnsw->entryLevel = this->local.l;
		}
	}

	size_t DebugGraph::getEnterPoint() {
		return this->hnsw->entryIdx;
	}

	void GraphDebugWrapper::insert(float* data, size_t idx) {
		this->debugObj->directInsert(data, idx);
	}

	GraphDebugWrapper::~GraphDebugWrapper() {
		delete this->debugObj;
	}

	GraphDebugWrapper::GraphDebugWrapper(const HNSWConfigPtr& cfg) : GraphWrapper(cfg, "chm-HNSW-Debug"), debugObj(nullptr) {}

	DebugHNSW* GraphDebugWrapper::getDebugObject() {
		return this->debugObj;
	}

	void GraphDebugWrapper::init() {
		GraphWrapper::init();
		this->debugObj = new DebugGraph(this->hnsw);
	}
}
