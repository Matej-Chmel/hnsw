#define CHM_HNSW_INTERMEDIATE
#define DECIDE_BY_IDX
#include <cstdlib>
#include "Runner/HnswRunner.hpp"
namespace fs = chm::fs;

constexpr auto CHECK_INTERMEDIATES = false;
constexpr size_t DIM = 128;
constexpr size_t EF_CONSTRUCTION = 200;
constexpr auto ELEM_MAX = 1.f;
constexpr auto ELEM_MIN = 0.f;
constexpr size_t K = 10;
constexpr size_t M = 16;
constexpr size_t NODE_COUNT = 100000;
constexpr size_t NODE_SEED = 1000;
constexpr auto QUERY_COUNT = std::max(1ULL, NODE_COUNT / 100);
constexpr auto QUERY_SEED = NODE_SEED + 1;
constexpr auto REF_ALGO = chm::HnswKind::CHM_AUTO;
constexpr auto SUB_ALGO = chm::HnswKind::CHM_AUTO;
constexpr auto USE_EUCLID = true;
constexpr auto USE_SIFT = false;

template<typename Coord>
chm::ICoordsPtr<Coord> getNodes(const fs::path& datasetsDir) {
	if constexpr(USE_SIFT)
		return std::make_shared<chm::ReadCoords<Coord>>(datasetsDir / "sift1M.bin");
	return std::make_shared<chm::RndCoords<Coord>>(NODE_COUNT, DIM, ELEM_MIN, ELEM_MAX, NODE_SEED);
}

template<typename Coord>
chm::ICoordsPtr<Coord> getQueries(const fs::path& datasetsDir) {
	if constexpr(USE_SIFT)
		return std::make_shared<chm::ReadCoords<Coord>>(datasetsDir / "siftQ1M.bin");
	return std::make_shared<chm::RndCoords<Coord>>(QUERY_COUNT, DIM, ELEM_MIN, ELEM_MAX, QUERY_SEED);
}

template<typename Coord>
chm::FoundNeighborsPtr<Coord> getTrueNeighbors(
	const fs::path& datasetsDir, const chm::ICoordsPtr<Coord>& nodes, const chm::ICoordsPtr<Coord>& queries
) {
	if constexpr(USE_SIFT)
		return chm::readTrueNeighbors<Coord>(datasetsDir / "knnQA1M.bin", K, 100);
	return chm::bruteforce<Coord, USE_EUCLID>(nodes->get(), queries->get(), DIM, K);
}

template<typename Coord>
void run() {
	using namespace chm;

	const auto slnDir = fs::path(SOLUTION_DIR);
	const auto datasetsDir = slnDir / "datasets";
	const auto outDir = slnDir / "logs";

	if(!fs::exists(outDir))
		fs::create_directories(outDir);

	const auto nodes = getNodes<Coord>(datasetsDir);
	const auto cfg = std::make_shared<HnswCfg>(DIM, EF_CONSTRUCTION, M, nodes->getCount(DIM), NODE_SEED, USE_EUCLID);
	const auto runner = createRunner<Coord>(
		CHECK_INTERMEDIATES, nodes,
		std::make_shared<HnswRunCfg>(
			std::make_shared<HnswType>(
				cfg, CHECK_INTERMEDIATES, REF_ALGO, std::make_shared<HnswSettings>(false, true, false)
			),
			std::make_shared<HnswType>(
				cfg, CHECK_INTERMEDIATES, SUB_ALGO, std::make_shared<HnswSettings>(false, true, true)
			)
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

	const auto queries = getQueries<Coord>(datasetsDir);
	std::ofstream stream(outDir / "search.log");
	const auto trueNeighbors = getTrueNeighbors<Coord>(datasetsDir, nodes, queries);

	for(size_t i = 0; i < efsLen; i++) {
		const auto searchCfg = std::make_shared<SearchCfg<Coord>>(queries, efs[i], K);
		const auto searchRes = runner->search(searchCfg, trueNeighbors);

		searchCfg->print(stream);
		searchRes->print(stream);

		if(i != efsLastIdx)
			stream << '\n';
	}
}

int main() {
	try {
		run<float>();
		return EXIT_SUCCESS;

	} catch(const std::runtime_error& e) {
		std::cout << e.what() << '\n';
	}

	return EXIT_FAILURE;
}
