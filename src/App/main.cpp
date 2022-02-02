#define CHM_HNSW_INTERMEDIATE
#define DECIDE_BY_IDX
#include <cstdlib>
#include "Runner/HnswRunner.hpp"

constexpr auto CHECK_INTERMEDIATES = false;
constexpr size_t DIM = 128;
constexpr size_t EF_CONSTRUCTION = 200;
constexpr float ELEM_MAX = 1.f;
constexpr float ELEM_MIN = 0.f;
constexpr size_t K = 10;
constexpr size_t M = 16;
constexpr size_t NODE_COUNT = 3000;
constexpr size_t NODE_SEED = 100;
constexpr size_t QUERY_COUNT = 10;
constexpr auto QUERY_SEED = NODE_SEED + 1;
constexpr auto REF_ALGO = chm::HnswKind::HNSWLIB;
constexpr auto SUB_ALGO = chm::HnswKind::CHM_AUTO;
constexpr auto USE_EUCLID = true;

int main() {
	try {
		using namespace chm;

		const auto slnDir = fs::path(SOLUTION_DIR);
		const auto outDir = slnDir / "logs";

		if(!fs::exists(outDir))
			fs::create_directories(outDir);

		const auto nodes = std::make_shared<RndCoords<float>>(NODE_COUNT, DIM, ELEM_MIN, ELEM_MAX, NODE_SEED);
		// const auto coords = std::make_shared<ReadCoords<float>>(slnDir / "datasets" / "sift1M.bin", NODE_COUNT, DIM);

		const auto runner = createRunner<float>(
			std::make_shared<HnswCfg>(DIM, EF_CONSTRUCTION, M, NODE_COUNT, NODE_SEED, USE_EUCLID),
			CHECK_INTERMEDIATES, nodes,
			std::make_shared<HnswRunCfg>(
				std::make_shared<HnswType>(CHECK_INTERMEDIATES, REF_ALGO),
				std::make_shared<HnswType>(CHECK_INTERMEDIATES, SUB_ALGO)
			)
		);

		const auto buildRes = runner->build();
		buildRes->print(std::cout);
		buildRes->write(outDir);
		std::cout << '\n';

		std::vector<size_t> efs;

		for(size_t i = 10; i < 30; i++)
			efs.push_back(i);
		for(size_t i = 100; i < 2000; i += 100)
			efs.push_back(i);

		const auto efsLen = efs.size();
		const auto efsLastIdx = efsLen - 1;

		const auto queries = std::make_shared<RndCoords<float>>(QUERY_COUNT, DIM, ELEM_MIN, ELEM_MAX, QUERY_SEED);
		std::ofstream stream(outDir / "search.log");
		const auto trueNeighbors = bruteforce(nodes->get(), queries->get(), DIM, K);

		for(size_t i = 0; i < efsLen; i++) {
			const auto searchCfg = std::make_shared<SearchCfg<float>>(queries, efs[i], K);
			const auto searchRes = runner->search(searchCfg, trueNeighbors);

			searchCfg->print(stream);
			searchRes->print(stream);

			if(i != efsLastIdx)
				stream << '\n';
		}

		return EXIT_SUCCESS;

	} catch(const std::runtime_error& e) {
		std::cout << e.what() << '\n';
	}

	return EXIT_FAILURE;
}
