#include "KNNAlgorithm.hpp"

namespace chm {
	KNNAlgorithm::KNNAlgorithm(const std::string& info) : info(info) {}

	std::string KNNAlgorithm::getInfo() const {
		return this->info;
	}

	TrueKNNAlgorithm::TrueKNNAlgorithm(const std::string& info) : KNNAlgorithm(info) {}

	void KNNResult::resize(size_t queryCount) {
		this->distances.resize(queryCount);
		this->indices.resize(queryCount);
	}

	HNSWConfig::HNSWConfig::HNSWConfig(size_t dim, size_t efConstruction, size_t M, size_t maxElements, size_t seed)
		: dim(dim), efConstruction(efConstruction), M(M), maxElements(maxElements), seed(seed) {}

	size_t HNSWAlgorithm::getElementCount(const FloatVecPtr& coords) const {
		return coords->size() / this->cfg->dim;
	}

	HNSWAlgorithm::HNSWAlgorithm(const HNSWConfigPtr& cfg, const std::string& info) : KNNAlgorithm(info), cfg(cfg) {}
}
