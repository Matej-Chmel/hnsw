#include "DebugHNSW.hpp"

namespace chm {
	Node::Node() : distance(0.f), idx(0) {}

	Node::Node(float distance, size_t idx) : distance(distance), idx(idx) {}

	bool NodeComparator::operator()(const Node& a, const Node& b) const {
		return a.distance < b.distance;
	}
}
