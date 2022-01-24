#include <cstdlib>
#include "Runner/HnswRunner.hpp"

constexpr size_t DIM = 128;
constexpr size_t NODE_COUNT = 1000;
constexpr auto REF_INTERMEDIATE = true;
constexpr auto RUN_INTERMEDIATE = true;
constexpr auto SUB_INTERMEDIATE = true;
constexpr size_t SEED = 100;

int main() {
	using namespace chm;

	try {
		createRunner<float>(
			std::make_shared<HnswCfg>(DIM, 200, 16, NODE_COUNT, SEED, true),
			std::make_shared<RndCoords<float>>(NODE_COUNT, DIM, 0.f, 1.f, SEED),
			RUN_INTERMEDIATE,
			std::make_shared<HnswRunCfg>(
				std::make_shared<HnswType>(REF_INTERMEDIATE, HnswKind::HNSWLIB),
				std::make_shared<HnswType>(SUB_INTERMEDIATE, HnswKind::CHM_AUTO)
				)
		)->run()->print(std::cout);
	} catch(const std::runtime_error& e) {
		std::cout << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
