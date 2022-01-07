#include "refImplWrappers.hpp"

namespace chm {
	const std::string BacaWrapper::NAME = "Baca-HNSW";

	BacaWrapper::BacaWrapper(const HNSWConfigPtr& cfg) : HNSWAlgo(cfg, BacaWrapper::NAME), ef(0), hnsw(nullptr) {}

	void BacaWrapper::init() {
		this->hnsw = new HNSW(int(this->cfg->M), int(this->cfg->M), int(this->cfg->efConstruction));
		this->hnsw->init(uint32_t(this->cfg->dim), uint32_t(this->cfg->maxElements));
	}

	void BacaWrapper::insert(float* data, size_t) {
		this->hnsw->insert(data);
	}

	BacaWrapper::~BacaWrapper() {
		delete this->hnsw;
	}

	IdxVec3DPtr BacaWrapper::getConnections() const {
		const auto layersLen = this->hnsw->layers_.size();
		auto res = std::make_shared<IdxVec3D>(this->hnsw->actual_node_count_ + 1);
		auto& r = *res;

		for(size_t hnswLayerIdx = 0; hnswLayerIdx < layersLen; hnswLayerIdx++) {
			const auto expectedNodeLayersLen = hnswLayerIdx + 1;
			const auto& hnswLayer = this->hnsw->layers_[hnswLayerIdx];

			for(const auto& n : hnswLayer->nodes) {
				auto& nodeLayers = r[n->order];

				if(nodeLayers.size() < expectedNodeLayersLen)
					nodeLayers.resize(expectedNodeLayersLen);

				auto& nodeLayer = nodeLayers[hnswLayerIdx];
				nodeLayer.reserve(n->neighbors.size());

				for(const auto& ne : n->neighbors)
					nodeLayer.push_back(ne.node_order);
			}
		}

		chm::sortConnections(res);
		return res;
	}

	KNNResultPtr BacaWrapper::search(const FloatVecPtr& coords, size_t K) {
		auto& c = *coords;
		const auto count = this->getElementCount(coords);
		auto res = std::make_shared<KNNResult>();
		auto& r = *res;

		r.resize(count);

		for(size_t i = 0; i < count; i++) {
			auto& distances = r.distances[i];
			auto& IDs = r.indices[i];

			this->hnsw->aproximateKnn(&c[i * this->cfg->dim], int(K), int(this->ef));

			auto len = std::min(this->hnsw->W_.size(), K);
			distances.reserve(len);
			IDs.reserve(len);

			for(size_t WIdx = 0; WIdx < len; WIdx++) {
				auto& node = this->hnsw->W_[i];
				distances.push_back(node.distance);
				IDs.push_back(node.node_order);
			}
		}

		return res;
	}

	void BacaWrapper::setSearchEF(size_t ef) {
		this->ef = ef;
	}

	const std::string hnswlibWrapper::NAME = "hnswlib-HNSW";

	void hnswlibWrapper::init() {
		this->space = new hnswlib::L2Space(this->cfg->dim);
		this->hnsw = new hnswlib::HierarchicalNSW(this->space, this->cfg->maxElements, this->cfg->M, this->cfg->efConstruction, this->cfg->seed);
	}

	void hnswlibWrapper::insert(float* data, size_t idx) {
		this->hnsw->addPoint(data, idx);
	}

	hnswlibWrapper::~hnswlibWrapper() {
		delete this->hnsw;
		delete this->space;
	}

	IdxVec3DPtr hnswlibWrapper::getConnections() const {
		auto res = std::make_shared<IdxVec3D>(this->hnsw->cur_element_count);
		auto& r = *res;

		for(size_t i = 0; i < this->hnsw->cur_element_count; i++) {
			auto& nodeLayers = r[i];
			const auto nodeLayersLen = size_t(this->hnsw->element_levels_[i]) + 1;
			nodeLayers.resize(nodeLayersLen);

			for(size_t level = 0; level < nodeLayersLen; level++) {
				const auto& linkList = this->hnsw->get_linklist_at_level(hnswlib::tableint(i), int(level));
				const auto linkListLen = this->hnsw->getListCount(linkList);
				auto& neighbors = nodeLayers[level];
				neighbors.reserve(linkListLen);

				for(size_t i = 1; i <= linkListLen; i++)
					neighbors.push_back(linkList[i]);
			}
		}

		chm::sortConnections(res);
		return res;
	}

	hnswlibWrapper::hnswlibWrapper(const HNSWConfigPtr& cfg) : HNSWAlgo(cfg, hnswlibWrapper::NAME), hnsw(nullptr), space(nullptr) {}

	KNNResultPtr hnswlibWrapper::search(const FloatVecPtr& coords, size_t K) {
		const auto& c = *coords;
		const auto count = this->getElementCount(coords);
		auto res = std::make_shared<KNNResult>();
		auto& r = *res;

		r.resize(count);

		for(size_t i = 0; i < count; i++) {
			auto results = this->hnsw->searchKnn(&c[i * this->cfg->dim], K);
			const auto resultsLen = results.size();

			auto& distances = r.distances[i];
			auto& IDs = r.indices[i];
			distances.reserve(resultsLen);
			IDs.reserve(resultsLen);

			while(!results.empty()) {
				auto& item = results.top();
				distances.push_back(item.first);
				IDs.push_back(item.second);
				results.pop();
			}
		}

		return res;
	}

	void hnswlibWrapper::setSearchEF(size_t ef) {
		this->hnsw->setEf(ef);
	}
}
