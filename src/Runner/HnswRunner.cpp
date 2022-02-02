#include "HnswRunner.hpp"

namespace chm {
	HnswRunCfg::HnswRunCfg(const HnswTypePtr& refType, const HnswTypePtr& subType) : refType(refType), subType(subType) {}

	void Timer::start() {
		this->from = chr::steady_clock::now();
	}

	chr::microseconds Timer::stop() {
		return chr::duration_cast<chr::microseconds>(chr::steady_clock::now() - this->from);
	}

	Timer::Timer() : from{} {}

	IdxVec3DPtr sortedInPlace(const IdxVec3DPtr& conn) {
		for(auto& nodeLayers : *conn)
			for(auto& layer : nodeLayers)
				std::sort(layer.begin(), layer.end());
		return conn;
	}
}
