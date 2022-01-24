#include <cstdlib>
#include "Runner/HnswRunner.hpp"

constexpr size_t DIM = 128;
constexpr size_t NODE_COUNT = 5000;
constexpr auto REF_INTERMEDIATE = false;
constexpr auto RUN_INTERMEDIATE = false;
constexpr auto SUB_INTERMEDIATE = false;
constexpr size_t SEED = 100;

int main() {
	using namespace chm;

	const auto outDir = fs::path(SOLUTION_DIR) / "logs";

	if(!fs::exists(outDir))
		fs::create_directories(outDir);

	try {
		const auto res = createRunner<float>(
			std::make_shared<HnswCfg>(DIM, 200, 16, NODE_COUNT, SEED, true),
			std::make_shared<RndCoords<float>>(NODE_COUNT, DIM, 0.f, 1.f, SEED),
			RUN_INTERMEDIATE,
			std::make_shared<HnswRunCfg>(
				std::make_shared<HnswType>(REF_INTERMEDIATE, HnswKind::HNSWLIB),
				std::make_shared<HnswType>(SUB_INTERMEDIATE, HnswKind::CHM_AUTO)
				)
		)->run();

		res->print(std::cout);
		res->write(outDir);

	} catch(const std::runtime_error& e) {
		std::cout << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
