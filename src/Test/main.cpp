#include "Runner.hpp"

constexpr size_t EF_CONSTRUCTION = 200;
constexpr auto DEBUG_BUILD = false;
constexpr size_t DIM = 128;
constexpr auto ELEMENT_MAX = 1.f;
constexpr auto ELEMENT_MIN = 0.f;
constexpr unsigned int ELEMENT_SEED = 100;
constexpr unsigned int LEVEL_SEED = 100;
constexpr size_t MAX_NEIGHBORS = 16;
constexpr size_t NODE_COUNT = 10;
constexpr auto TRACK = false;

int main() {
	return chm::Runner(
		{
			chm::HNSWAlgoKind::HNSWLIB, chm::HNSWAlgoKind::CHM_HNSW
		},
		std::make_shared<chm::HNSWConfig>(DIM, EF_CONSTRUCTION, MAX_NEIGHBORS, NODE_COUNT, LEVEL_SEED),
		DEBUG_BUILD,
		std::make_shared<chm::ElementGen>(NODE_COUNT, DIM, ELEMENT_MIN, ELEMENT_MAX, ELEMENT_SEED),
		std::cout,
		TRACK
	).runAll();
}
