#include "hnswFactoryMethods.hpp"

namespace chm {
	HnswType::HnswType(const HnswCfgPtr& cfg, const bool isIntermediate, const HnswKind kind, const HnswSettingsPtr& settings)
		: cfg(cfg), isIntermediate(isIntermediate), kind(kind), settings(settings) {}
}
