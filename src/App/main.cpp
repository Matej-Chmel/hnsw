#define DECIDE_BY_IDX
#include <cstdlib>
#include "Runner/HnswRunner.hpp"

constexpr size_t DIM = 128;
constexpr auto INTERMEDIATE = false;
constexpr size_t NODE_COUNT = 1000000;
constexpr size_t SEED = 100;

int main() {
	using namespace chm;

	const auto slnDir = fs::path(SOLUTION_DIR);
	const auto outDir = slnDir / "logs";

	if(!fs::exists(outDir))
		fs::create_directories(outDir);

	// const auto coords = std::make_shared<RndCoords<float>>(NODE_COUNT, DIM, 0.f, 1.f, SEED);
	const auto coords = std::make_shared<ReadCoords<float>>(slnDir / "datasets" / "sift1M.bin", NODE_COUNT, DIM);

	try {
		const auto res = createRunner<float>(
			std::make_shared<HnswCfg>(DIM, 200, 16, NODE_COUNT, SEED, true),
			coords, INTERMEDIATE,
			std::make_shared<HnswRunCfg>(
				std::make_shared<HnswType>(INTERMEDIATE, HnswKind::HNSWLIB),
				std::make_shared<HnswType>(INTERMEDIATE, HnswKind::CHM_AUTO)
			)
		)->run();

		res->print(std::cout, INTERMEDIATE);
		res->write(outDir, INTERMEDIATE);

	} catch(const std::runtime_error& e) {
		std::cout << e.what() << '\n';
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}
