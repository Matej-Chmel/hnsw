#include <algorithm>
#include <cmath>
#include <unordered_set>
#include "Graph.hpp"

namespace chm {
	void Config::calcML() {
		this->mL = 1.0 / std::log(1.0 * this->M);
	}

	Config::Config(const HNSWConfigPtr& cfg)
		: dim(cfg->dim), maxNodeCount(cfg->maxElements), efConstruction(cfg->efConstruction), M(cfg->M), Mmax(cfg->M), Mmax0(cfg->M * 2),
		useHeuristic(true), extendCandidates(false), keepPrunedConnections(false) {

		this->calcML();
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

	void DynamicList::add(float distance, size_t nodeID) {
		this->furthestHeap.push(distance, nodeID);
		this->nearestHeap.push(distance, nodeID);
	}

	void DynamicList::clear() {
		this->furthestHeap.clear();
		this->nearestHeap.clear();
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

	float Graph::getDistance(const float* node, const float* query, bool useCache, size_t nodeID) {
		if(useCache) {
			auto iter = this->distancesCache.find(nodeID);

			if(iter != this->distancesCache.end())
				return iter->second;
		}

		float res = 0.f;

		for(size_t i = 0; i < this->cfg.dim; i++) {
			float diff = node[i] - query[i];
			res += diff * diff;
		}

		if(useCache)
			this->distancesCache[nodeID] = res;

		return res;
	}

	size_t Graph::getNewLevel() {
		std::uniform_real_distribution<double> dist(0.0, 1.0);

		return size_t(
			-std::log(dist(this->gen)) * this->cfg.mL
		);
	}

	void Graph::insert(const float* coords, size_t queryID) {
		std::copy(coords, coords + cfg.dim, this->coords.begin() + this->nodeCount * this->cfg.dim);

		if(queryID == 0) {
			this->entryLevel = this->getNewLevel();
			this->nodeCount = 1;
			this->initLayers(this->entryID, this->entryLevel);
			return;
		}

		this->distancesCache.clear();
		this->nodeCount++;

		DynamicList W(this->getDistance(this->getCoords(this->entryID), coords, true, this->entryID), this->entryID);
		auto L = this->entryLevel;
		auto l = this->getNewLevel();

		this->initLayers(queryID, l);

		for(size_t lc = L; lc > l; lc--) {
			this->searchLayer(coords, W, 1, lc);
			W.keepOnlyNearest();
		}

		for(size_t lc = std::min(L, l);; lc--) {
			size_t layerMmax = lc == 0 ? this->cfg.Mmax0 : this->cfg.Mmax;

			this->searchLayer(coords, W, this->cfg.efConstruction, lc);

			NearestHeap neighbors(W.nearestHeap);
			this->selectNeighbors(coords, neighbors, this->cfg.M, lc, true);
			this->connect(queryID, neighbors, lc);

			for(auto& e : neighbors.nodes) {
				auto& eConn = this->layers[e.nodeID][lc];

				if(eConn.size() > layerMmax) {
					auto eCoords = this->getCoords(e.nodeID);

					NearestHeap eNewConn;
					this->fillHeap(eCoords, eConn, eNewConn);
					this->selectNeighbors(eCoords, eNewConn, layerMmax, lc, false);
					eNewConn.fillLayer(eConn);
				}
			}

			if(lc == 0)
				break;

			W.clear();
			auto& nearestNeighbor = neighbors.top();
			W.add(nearestNeighbor.distance, nearestNeighbor.nodeID);
		}

		if(l > L) {
			this->entryID = queryID;
			this->entryLevel = l;
		}
	}

	void Graph::searchLayer(const float* query, DynamicList& W, size_t ef, size_t lc) {
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
					float eDistance = this->getDistance(this->getCoords(eID), query, true, eID);

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

	void Graph::selectNeighbors(const float* query, NearestHeap& outC, size_t M, size_t lc, bool useCache) {
		if(this->cfg.useHeuristic)
			this->selectNeighborsHeuristic(query, outC, M, lc, useCache);
		else
			this->selectNeighborsSimple(outC, M);
	}

	void Graph::selectNeighborsHeuristic(const float* query, NearestHeap& outC, size_t M, size_t lc, bool useCache) {
		if(outC.size() < M)
			return;

		NearestHeap R;
		auto& W = outC;

		/*
		if(this->cfg.extendCandidates) {
			std::unordered_set<size_t> visited;

			for(auto& e : outC.nodes) {
				auto& neighbors = this->layers[e.nodeID][lc];

				for(auto& eAdjID : neighbors)
					if(visited.find(eAdjID) == visited.end())
						visited.insert(eAdjID);
			}

			for(const auto& ID : visited)
				W.push(this->getDistance(this->getCoords(ID), query, useCache, ID), ID);
		}
		*/

		// NearestHeap Wd;

		while(W.size() > 0 && R.size() < M) {
			auto e = W.pop();

			for(auto& rNode : R.nodes)
				if(this->getDistance(this->getCoords(e.nodeID), this->getCoords(rNode.nodeID)) < e.distance)
					goto isNotCloser;

			R.push(e.distance, e.nodeID);

			/*
			else if(this->cfg.keepPrunedConnections)
				Wd.push(e.distance, e.nodeID);
			*/

			isNotCloser:;
		}

		/*
		if(this->cfg.keepPrunedConnections) {
			while(Wd.size() > 0 && R.size() < M) {
				auto discardedNearest = Wd.pop();
				R.push(discardedNearest.distance, discardedNearest.nodeID);
			}
		}
		*/

		outC.swap(R);
	}

	void Graph::selectNeighborsSimple(NearestHeap& outC, size_t M) {
		outC.keepNearest(M);
	}

	void Graph::knnSearch(const float* query, size_t K, size_t ef, IdxVec& outIDs, FloatVec& outDistances) {
		this->distancesCache.clear();

		DynamicList W(this->getDistance(this->getCoords(this->entryID), query, true, this->entryID), this->entryID);
		auto L = this->entryLevel;

		for(size_t lc = L; lc > 0; lc--) {
			this->searchLayer(query, W, 1, lc);
			W.keepOnlyNearest();
		}

		this->searchLayer(query, W, ef, 0);
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

	void Graph::fillHeap(const float* query, IdxVec& eConn, NearestHeap& eNewConn) {
		eNewConn.reserve(eConn.size());

		for(auto& ID : eConn)
			eNewConn.push(this->getDistance(this->getCoords(ID), query), ID);
	}

	void Graph::initLayers(size_t queryID, size_t level) {
		size_t nLayers = level + 1;
		auto& qLayers = this->layers[queryID];
		qLayers.resize(nLayers);
	}

	Graph::Graph(const Config& cfg, unsigned int seed)
		: cfg(cfg), entryID(0), entryLevel(0), debugStream(nullptr), nodeCount(0) {

		this->coords.resize(cfg.dim * cfg.maxNodeCount);
		this->gen.seed(seed);
		this->layers.resize(cfg.maxNodeCount);
	}

	void Graph::build(const FloatVec& coords) {
		auto count = coords.size() / this->cfg.dim;
		auto lastID = count - 1;

		for(size_t i = 0; i < count; i++) {
			this->insert(&coords[i * this->cfg.dim], i);

			if(this->debugStream) {
				this->printLayers(*this->debugStream);

				if(i != lastID)
					*this->debugStream << '\n';
			}
		}
	}

	void Graph::search(const FloatVec& queryCoords, size_t K, size_t ef, IdxVec2D& outIDs, FloatVec2D& outDistances) {
		const auto count = queryCoords.size() / this->cfg.dim;

		if(!this->nodeCount)
			throw AppError("No elements in the graph. Build the graph before searching.");

		outIDs.resize(count);
		outDistances.resize(count);

		for(size_t i = 0; i < count; i++)
			this->knnSearch(&queryCoords[i * this->cfg.dim], K, ef, outIDs[i], outDistances[i]);
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

	const float* Graph::getCoords(size_t idx) {
		return &this->coords[idx * this->cfg.dim];
	}

	void GraphWrapper::insert(float* data, size_t idx) {
		this->hnsw->insert(data, idx);
	}

	GraphWrapper::~GraphWrapper() {
		delete this->hnsw;
	}

	IdxVec3DPtr GraphWrapper::getConnections() const {
		auto res = std::make_shared<IdxVec3D>(this->hnsw->layers.begin(), this->hnsw->layers.begin() + this->hnsw->getNodeCount());

		for(auto& nodeLayers : *res)
			for(auto& layer : nodeLayers)
				std::sort(layer.begin(), layer.end());

		return res;
	}

	DebugHNSW* GraphWrapper::getDebugObject() {
		return nullptr;
	}

	GraphWrapper::GraphWrapper(const HNSWConfigPtr& cfg) : HNSWAlgo(cfg, "chm-HNSW"), ef(0), hnsw(nullptr) {}

	void GraphWrapper::init() {
		this->hnsw = new Graph(Config(this->cfg), this->cfg->seed);
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
}
