#define DECIDE_BY_IDX
#include <cstdlib>
#include "Runner/HnswRunner.hpp"

constexpr auto CHECK_INTERMEDIATES = true;
constexpr size_t DIM = 128;
constexpr size_t EF_CONSTRUCTION = 200;
constexpr size_t M = 16;
constexpr size_t NODE_COUNT = 100000;
constexpr auto REF_ALGO = chm::HnswKind::HNSWLIB;
constexpr size_t SEED = 100;
constexpr auto SUB_ALGO = chm::HnswKind::CHM_AUTO;
constexpr auto USE_EUCLID = true;

int main() {
	try {
		using namespace chm;

		const auto slnDir = fs::path(SOLUTION_DIR);
		const auto outDir = slnDir / "logs";

		if(!fs::exists(outDir))
			fs::create_directories(outDir);

		const auto coords = std::make_shared<RndCoords<float>>(NODE_COUNT, DIM, 0.f, 1.f, SEED);
		// const auto coords = std::make_shared<ReadCoords<float>>(slnDir / "datasets" / "sift1M.bin", NODE_COUNT, DIM);

		const auto res = createRunner<float>(
			std::make_shared<HnswCfg>(DIM, EF_CONSTRUCTION, M, NODE_COUNT, SEED, USE_EUCLID),
			CHECK_INTERMEDIATES, coords,
			std::make_shared<HnswRunCfg>(
				std::make_shared<HnswType>(CHECK_INTERMEDIATES, REF_ALGO),
				std::make_shared<HnswType>(CHECK_INTERMEDIATES, SUB_ALGO)
			)
		)->run();

		res->print(std::cout);
		res->write(outDir);
		return EXIT_SUCCESS;

	} catch(const std::runtime_error& e) {
		std::cout << e.what() << '\n';
	}

	return EXIT_FAILURE;
}
