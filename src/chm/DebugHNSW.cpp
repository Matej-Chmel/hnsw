#include "DebugHNSW.hpp"

namespace chm {
	Node::Node() : distance(0.f), idx(0) {}

	Node::Node(float distance, size_t idx) : distance(distance), idx(idx) {}

	bool NodeComparator::operator()(const Node& a, const Node& b) const {
		if(a.distance == b.distance)
			return a.idx < b.idx;

		return a.distance < b.distance;
	}

	void DebugHNSW::directInsert(float* data, size_t idx) {
		this->startInsert(data, idx);
		this->prepareUpperSearch();

		auto range = this->getUpperRange();

		if (range.shouldLoop)
			for (auto lc = range.start; lc > range.end; lc--)
				this->searchUpperLayers(lc);

		range = this->getLowerRange();

		if (range.shouldLoop) {
			this->prepareLowerSearch();

			for (auto lc = range.start;; lc--) {
				this->searchLowerLayers(lc);
				this->selectOriginalNeighbors(lc);
				this->connect(lc);
				this->prepareNextLayer(lc);

				if (lc == 0)
					break;
			}
		}

		this->setupEnterPoint();
	}
}
