#include "Runner/HnswRunner.hpp"

constexpr size_t DIM = 128;
constexpr size_t NODE_COUNT = 35000;
constexpr size_t SEED = 100;

int main() {
	using namespace chm;

	HnswRunner<float>(
		std::make_shared<HnswCfg>(DIM, 200, 16, NODE_COUNT, SEED, true),
		std::make_shared<RndCoords<float>>(NODE_COUNT, DIM, 0.f, 1.f, SEED),
		std::make_shared<HnswRunCfg>(HnswKind::HNSWLIB, HnswKind::CHM_AUTO)
	).run()->print(std::cout);

	return 0;
}
