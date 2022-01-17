#include <algorithm>
#include <cmath>
#include <unordered_set>
#include "Graph.hpp"

namespace chm {
	void Config::calcML() {
		this->mL = 1 / std::logf(float(this->M));
	}

	Config::Config(const HNSWConfigPtr& cfg)
		: dim(cfg->dim), efConstruction(cfg->efConstruction), M(cfg->M), Mmax(cfg->M), Mmax0(cfg->M * 2),
		useHeuristic(true), extendCandidates(false), keepPrunedConnections(false) {

		this->calcML();
	}

	Config::Config(size_t dim)
		: dim(dim), efConstruction(200), M(16), Mmax(this->M), Mmax0(this->M * 2),
		useHeuristic(false), extendCandidates(false), keepPrunedConnections(false) {

		this->calcML();
	}

	Config& Config::setHeuristic(bool extendCandidates, bool keepPrunedConnections) {
		this->useHeuristic = true;
		this->extendCandidates = extendCandidates;
		this->keepPrunedConnections = keepPrunedConnections;
		return *this;
	}

	constexpr bool FurthestComparator::operator()(const NodeDistance& a, const NodeDistance& b) const noexcept {
		return a.distance < b.distance;
	}

	void FurthestHeap::clear() {
		this->nodes.clear();
	}

	FurthestHeap::FurthestHeap() {}

	FurthestHeap::FurthestHeap(NodeDistanceVec& ep) {
		this->nodes = ep;
		std::make_heap(this->nodes.begin(), this->nodes.end(), this->cmp);
	}

	void FurthestHeap::push(float distance, size_t nodeID) {
		this->nodes.push_back({distance, nodeID});
		std::push_heap(this->nodes.begin(), this->nodes.end(), this->cmp);
	}

	NodeDistance FurthestHeap::pop() {
		std::pop_heap(this->nodes.begin(), this->nodes.end(), this->cmp);
		NodeDistance item = this->nodes.back();
		this->nodes.pop_back();
		return item;
	}

	NodeDistance FurthestHeap::top() {
		return this->nodes.front();
	}

	constexpr bool NearestComparator::operator()(const NodeDistance& a, const NodeDistance& b) const noexcept {
		return a.distance > b.distance;
	}

	void NearestHeap::clear() {
		this->nodes.clear();
	}

	void NearestHeap::fillLayer(IdxVec& layer) {
		layer.clear();
		layer.reserve(this->nodes.size());

		for(auto& item : this->nodes)
			layer.push_back(item.nodeID);
	}

	bool NearestHeap::isCloserThanAny(NodeDistance& node) {
		if(this->nodes.empty())
			return true;

		auto& nearest = this->top();
		return nearest.distance > node.distance;
	}

	void NearestHeap::keepNearest(size_t K) {
		if(this->size() > K) {
			std::vector<NodeDistance> nearest;
			nearest.reserve(K);

			for(size_t i = 0; i < K; i++)
				nearest.push_back(this->pop());

			nearest.swap(this->nodes);
		}
	}

	NearestHeap::NearestHeap() {}

	NearestHeap::NearestHeap(NearestHeap& other) {
		this->nodes = other.nodes;
	}

	void NearestHeap::push(float distance, size_t nodeID) {
		this->nodes.push_back({distance, nodeID});
		std::push_heap(this->nodes.begin(), this->nodes.end(), this->cmp);
	}

	NodeDistance NearestHeap::pop() {
		std::pop_heap(this->nodes.begin(), this->nodes.end(), this->cmp);
		NodeDistance item = this->nodes.back();
		this->nodes.pop_back();
		return item;
	}

	void NearestHeap::remove(size_t nodeID) {
		auto len = this->nodes.size();

		for(size_t i = 0; i < len; i++) {
			auto& item = this->nodes[i];

			if(item.nodeID == nodeID) {
				this->nodes.erase(this->nodes.begin() + i);
				std::make_heap(this->nodes.begin(), this->nodes.end(), this->cmp);
				break;
			}
		}
	}

	void NearestHeap::reserve(size_t s) {
		this->nodes.reserve(s);
	}

	size_t NearestHeap::size() {
		return this->nodes.size();
	}

	void NearestHeap::swap(NearestHeap& other) {
		this->nodes.swap(other.nodes);
	}

	NodeDistance& NearestHeap::top() {
		return this->nodes.front();
	}

	void DynamicList::clear() {
		this->furthestHeap.clear();
		this->nearestHeap.clear();
	}

	void DynamicList::add(float distance, size_t nodeID) {
		this->furthestHeap.push(distance, nodeID);
		this->nearestHeap.push(distance, nodeID);
	}

	DynamicList::DynamicList(float distance, size_t entryID) {
		this->add(distance, entryID);
	}

	void DynamicList::fillResults(size_t K, IdxVec& outIDs, FloatVec& outDistances) {
		size_t len = std::min(K, this->size());
		outIDs.reserve(len);
		outDistances.reserve(len);

		for(size_t i = 0; i < len; i++) {
			auto& item = this->nearestHeap.top();

			outIDs.push_back(item.nodeID);
			outDistances.push_back(item.distance);
			this->nearestHeap.pop();
		}
	}

	NodeDistance DynamicList::furthest() {
		return this->furthestHeap.top();
	}

	void DynamicList::keepOnlyNearest() {
		NodeDistance nearest = this->nearestHeap.top();

		this->clear();
		this->add(nearest.distance, nearest.nodeID);
	}

	void DynamicList::removeFurthest() {
		auto item = this->furthestHeap.pop();
		this->nearestHeap.remove(item.nodeID);
	}

	size_t DynamicList::size() {
		return this->nearestHeap.size();
	}

	float Graph::getDistance(size_t nodeID, size_t queryID, State s) {
		if(s != State::SHRINKING) {
			auto iter = this->distances.find(nodeID);

			if(iter != this->distances.end())
				return iter->second;
		}

		float* nodeCoords = this->coords + nodeID * this->cfg.dim;
		float* qCoords = (s == State::SEARCHING ? this->queryCoords : this->coords) + queryID * this->cfg.dim;

		float distance = 0.f;

		for(size_t i = 0; i < this->cfg.dim; i++) {
			float diff = nodeCoords[i] - qCoords[i];
			distance += diff * diff;
		}

		if(s != State::SHRINKING)
			this->distances[nodeID] = distance;

		return distance;
	}

	size_t Graph::getNewLevel() {
		return size_t(
			std::floorf(
				-std::logf(
					this->dist(this->gen) * this->cfg.mL
				)
			)
		);
	}

	void Graph::insert(size_t queryID) {
		DynamicList W(this->getDistance(this->entryID, queryID), this->entryID);
		auto L = this->entryLevel;
		auto l = this->getNewLevel();

		this->initLayers(queryID, l);

		for(size_t lc = L; lc > l; lc--) {
			this->searchLayer(queryID, W, 1, lc);
			W.keepOnlyNearest();
		}

		for(size_t lc = std::min(L, l);; lc--) {
			size_t layerMmax = lc == 0 ? this->cfg.Mmax0 : this->cfg.Mmax;

			this->searchLayer(queryID, W, this->cfg.efConstruction, lc);

			NearestHeap neighbors(W.nearestHeap);
			this->selectNeighbors(queryID, neighbors, this->cfg.M, lc);
			this->connect(queryID, neighbors, lc);

			for(auto& e : neighbors.nodes) {
				auto& eConn = this->layers[e.nodeID][lc];

				if(eConn.size() > layerMmax) {
					NearestHeap eNewConn;
					this->fillHeap(e.nodeID, eConn, eNewConn);

					this->selectNeighbors(e.nodeID, eNewConn, layerMmax, lc, State::SHRINKING);
					eNewConn.fillLayer(eConn);
				}
			}

			if(lc == 0)
				break;
		}

		if(l > L) {
			this->entryID = queryID;
			this->entryLevel = l;
		}
	}

	void Graph::searchLayer(size_t queryID, DynamicList& W, size_t ef, size_t lc, State s) {
		std::unordered_set<size_t> v;

		for(auto& item : W.nearestHeap.nodes)
			v.insert(item.nodeID);

		NearestHeap C(W.nearestHeap);

		while(C.size() > 0) {
			auto c = C.pop();
			auto f = W.furthest();

			if(c.distance > f.distance)
				break;

			auto& neighbors = this->layers[c.nodeID][lc];

			for(auto& eID : neighbors) {
				if(v.find(eID) == v.end()) {
					v.insert(eID);
					f = W.furthest();
					float eDistance = this->getDistance(eID, queryID, s);

					if(eDistance < f.distance || W.size() < ef) {
						C.push(eDistance, eID);
						W.add(eDistance, eID);

						if(W.size() > ef)
							W.removeFurthest();
					}
				}
			}
		}
	}

	void Graph::selectNeighbors(size_t queryID, NearestHeap& outC, size_t M, size_t lc, State s) {
		if(this->cfg.useHeuristic)
			this->selectNeighborsHeuristic(queryID, outC, M, lc, s);
		else
			this->selectNeighborsSimple(outC, M);
	}

	void Graph::selectNeighborsHeuristic(size_t queryID, NearestHeap& outC, size_t M, size_t lc, State s) {
		NearestHeap R;
		auto& W = outC;

		if(this->cfg.extendCandidates) {
			std::unordered_set<size_t> visited;

			for(auto& e : outC.nodes) {
				auto& neighbors = this->layers[e.nodeID][lc];

				for(auto& eAdjID : neighbors)
					if(visited.find(eAdjID) == visited.end())
						visited.insert(eAdjID);
			}

			for(const auto& ID : visited)
				W.push(this->getDistance(ID, queryID, s), ID);
		}

		NearestHeap Wd;

		while(W.size() > 0 && R.size() < M) {
			auto e = W.pop();

			if(R.isCloserThanAny(e))
				R.push(e.distance, e.nodeID);
			else if(this->cfg.keepPrunedConnections)
				Wd.push(e.distance, e.nodeID);
		}

		if(this->cfg.keepPrunedConnections) {
			while(Wd.size() > 0 && R.size() < M) {
				auto discardedNearest = Wd.pop();
				R.push(discardedNearest.distance, discardedNearest.nodeID);
			}
		}

		outC.swap(R);
	}

	void Graph::selectNeighborsSimple(NearestHeap& outC, size_t M) {
		outC.keepNearest(M);
	}

	void Graph::knnSearch(size_t queryID, size_t K, size_t ef, IdxVec& outIDs, FloatVec& outDistances) {
		DynamicList W(this->getDistance(this->entryID, queryID, State::SEARCHING), this->entryID);
		auto L = this->entryLevel;

		for(size_t lc = L; lc > 0; lc--) {
			this->searchLayer(queryID, W, 1, lc, State::SEARCHING);
			W.keepOnlyNearest();
		}

		this->searchLayer(queryID, W, ef, 0, State::SEARCHING);
		W.fillResults(K, outIDs, outDistances);
	}

	void Graph::connect(size_t queryID, NearestHeap& neighbors, size_t lc) {
		auto& qLayer = this->layers[queryID][lc];

		for(auto& item : neighbors.nodes) {
			if(queryID == item.nodeID)
				throw AppError("Tried to connect element "_f << queryID << " with itself at layer " << lc << '.');

			qLayer.push_back(item.nodeID);

			auto& itemLayer = this->layers[item.nodeID][lc];
			itemLayer.push_back(queryID);
		}
	}

	void Graph::fillHeap(size_t queryID, IdxVec& eConn, NearestHeap& eNewConn) {
		eNewConn.reserve(eConn.size());

		for(auto& ID : eConn)
			eNewConn.push(this->getDistance(ID, queryID, State::SHRINKING), ID);
	}

	void Graph::initLayers(size_t queryID, size_t level) {
		size_t nLayers = level + 1;
		auto& qLayers = this->layers[queryID];
		qLayers.resize(nLayers);
	}

	Graph::Graph(const Config& cfg, unsigned int seed, bool useRndSeed)
		: cfg(cfg), entryID(0), entryLevel(0), coords(nullptr), queryCoords(nullptr),
		gen(useRndSeed ? std::random_device{}() : seed),
		dist(0.f, 1.f), debugStream(nullptr), nodeCount(0) {}

	void Graph::build(float* coords, size_t count) {
		this->init(coords, count);
		auto lastID = count - 1;

		for(size_t i = 1; i < count; i++) {
			this->distances.clear();
			this->nodeCount++;
			this->insert(i);

			if(this->debugStream) {
				this->printLayers(*this->debugStream);

				if(i != lastID)
					*this->debugStream << '\n';
			}
		}
	}

	void Graph::search(
		float* queryCoords, size_t queryCount, size_t K, size_t ef,
		IdxVec2D& outIDs, FloatVec2D& outDistances) {

		if(!this->coords)
			throw AppError("No elements in the graph. Build the graph before searching.");

		outIDs.resize(queryCount);
		outDistances.resize(queryCount);
		this->queryCoords = queryCoords;

		for(size_t i = 0; i < queryCount; i++) {
			this->distances.clear();
			this->knnSearch(i, K, ef, outIDs[i], outDistances[i]);
		}
	}

	size_t Graph::getNodeCount() {
		return this->nodeCount;
	}

	void Graph::printLayers(std::ostream& s) {
		auto count = this->getNodeCount();
		auto lastID = count - 1;

		for(size_t nodeID = 0; nodeID < count; nodeID++) {
			s << "Node " << nodeID << '\n';

			auto& nodeLayers = this->layers[nodeID];
			auto nodeLayersLen = nodeLayers.size();

			for(size_t layerID = nodeLayersLen - 1;; layerID--) {
				s << "Layer " << layerID << ": ";

				auto& layer = nodeLayers[layerID];

				if(layer.empty()) {
					s << "EMPTY\n";
					continue;
				}

				auto lastIdx = layer.size() - 1;
				IdxVec sortedLayer(layer);
				std::sort(sortedLayer.begin(), sortedLayer.end());

				for(size_t i = 0; i < lastIdx; i++)
					s << sortedLayer[i] << ' ';

				s << sortedLayer[lastIdx] << '\n';

				if(layerID == 0)
					break;
			}

			if(nodeID != lastID)
				s << '\n';
		}
	}

	void Graph::setDebugStream(std::ostream& s) {
		this->debugStream = &s;
	}

	void Graph::init(float* coords, size_t count) {
		this->coords = coords;
		this->entryLevel = this->getNewLevel();
		this->nodeCount = 1;

		this->layers.resize(count);
		this->initLayers(this->entryID, this->entryLevel);
	}

	void GraphWrapper::insert(float* data, size_t idx) {
		if(idx == 0)
			return;

		this->hnsw->distances.clear();
		this->hnsw->nodeCount++;
		this->hnsw->insert(idx);
	}

	IdxVec3DPtr GraphWrapper::getConnections() const {
		return std::make_shared<IdxVec3D>(this->hnsw->layers.begin(), this->hnsw->layers.end());
	}

	DebugHNSW* GraphWrapper::getDebugObject() {
		return nullptr;
	}

	GraphWrapper::GraphWrapper(const HNSWConfigPtr& cfg) : HNSWAlgo(cfg, "chm-HNSW"), ef(0), hnsw(nullptr) {}

	void GraphWrapper::init() {
		this->hnsw = new Graph(Config(this->cfg), this->cfg->seed, false);
		this->hnsw->init();
	}

	void GraphWrapper::setSearchEF(size_t ef) {
		this->ef = ef;
	}
}
