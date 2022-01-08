#include "KNNAlgo.hpp"

namespace chm {
	KNNAlgo::KNNAlgo(const std::string& name) : name(name) {}

	std::string KNNAlgo::getName() const {
		return this->name;
	}

	TrueKNNAlgo::TrueKNNAlgo(const std::string& name) : KNNAlgo(name) {}

	void KNNResult::resize(size_t queryCount) {
		this->distances.resize(queryCount);
		this->indices.resize(queryCount);
	}

	HNSWConfig::HNSWConfig::HNSWConfig(size_t dim, size_t efConstruction, size_t M, size_t maxElements, unsigned int seed)
		: dim(dim), efConstruction(efConstruction), M(M), maxElements(maxElements), seed(seed) {}

	size_t HNSWAlgo::getElementCount(const FloatVecPtr& coords) const {
		return coords->size() / this->cfg->dim;
	}

	HNSWAlgo::HNSWAlgo(const HNSWConfigPtr& cfg, const std::string& name) : KNNAlgo(name), cfg(cfg) {}

	void HNSWAlgo::build(const FloatVecPtr& coords) {
		this->init();

		auto& c = *coords;
		const auto count = this->getElementCount(coords);

		for(size_t i = 0; i < count; i++)
			this->insert(&c[i * this->cfg->dim], i);
	}

	IdxVec3DPtr HNSWAlgo::buildAndTrack(const FloatVecPtr& coords, std::ostream& stream) {
		this->init();

		auto& c = *coords;
		const auto lastIdx = this->getElementCount(coords) - 1;

		for(size_t i = 0; i < lastIdx; i++) {
			this->insert(&c[i * this->cfg->dim], i);
			writeConnections(this->getConnections(), stream);
			stream << '\n';
		}

		this->insert(&c[lastIdx * this->cfg->dim], lastIdx);
		const auto finalConn = this->getConnections();
		writeConnections(finalConn, stream);
		return finalConn;
	}
}
