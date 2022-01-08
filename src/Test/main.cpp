#include "Runner.hpp"

constexpr size_t EF_CONSTRUCTION = 200;
constexpr size_t DIM = 128;
constexpr auto ELEMENT_MAX = 1.f;
constexpr auto ELEMENT_MIN = 0.f;
constexpr unsigned int ELEMENT_SEED = 100;
constexpr unsigned int LEVEL_SEED = 100;
constexpr size_t MAX_NEIGHBORS = 16;
constexpr size_t NODE_COUNT = 1000;

int main() {
	return chm::Runner(
		std::make_shared<chm::HNSWConfig>(DIM, EF_CONSTRUCTION, MAX_NEIGHBORS, NODE_COUNT, LEVEL_SEED),
		std::make_shared<chm::ElementGen>(NODE_COUNT, DIM, ELEMENT_MIN, ELEMENT_MAX, ELEMENT_SEED),
		std::cout
	).runAll();
}
