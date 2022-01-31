#define DECIDE_BY_IDX
#include <chrono>
#include <filesystem>
#include <hnswlib/hnswlib.h>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
namespace chr = std::chrono;
namespace fs = std::filesystem;

constexpr std::streamsize EF_W = 8;
constexpr std::streamsize ELAPSED_W = 36;
constexpr std::streamsize RECALL_W = 20;
constexpr auto USE_BRUTEFORCE = false;

using PriorityQueue = std::priority_queue<std::pair<float, hnswlib::labeltype>>;
using Answers = std::vector<PriorityQueue>;

class StopW {
	chr::steady_clock::time_point timeBegin;

public:
	chr::microseconds elapsedMicro() const {
		return chr::duration_cast<chr::microseconds>(chr::steady_clock::now() - this->timeBegin);
	}

	StopW() {
		this->timeBegin = chr::steady_clock::now();
	}
};

template<typename T>
long long convert(chr::microseconds& us) {
	const auto res = chr::duration_cast<T>(us);
	us -= chr::duration_cast<chr::microseconds>(res);
	return res.count();
}

void getGt(
	const std::vector<float>& mass, const size_t elemCount, const std::vector<float>& massQ, const size_t queryCount,
	const size_t dim, const size_t k, hnswlib::L2Space& space, Answers& answers
) {
	hnswlib::BruteforceSearch<float> bs(&space, elemCount);

	for(int i = 0; i < elemCount; i++)
		bs.addPoint(&mass[i * dim], size_t(i));

	(Answers(queryCount)).swap(answers);

	for(int i = 0; i < queryCount; i++)
		answers[i] = bs.searchKnn(&massQ[i * dim], k);
}

void getGt(
	const std::vector<float>& mass, const size_t elemCount, const std::vector<float>& massQ, const size_t queryCount,
	const std::vector<unsigned int>& massQA, const size_t dim, const size_t k, hnswlib::L2Space& space, Answers& answers
) {
	(Answers(queryCount)).swap(answers);

	const auto fstdistfunc_ = space.get_dist_func();

	std::cout << queryCount << "queries.\n";

	for(size_t i = 0; i < queryCount; i++)
		for(size_t j = 0; j < k; j++) {
			const auto correctIdx = massQA[i * 100 + j];
			answers[i].emplace(fstdistfunc_(&massQ[i * dim], &mass[correctIdx * dim], space.get_dist_func_param()), correctIdx);
		}
}

void printFill(std::ostream& s, const long long n, const size_t places = 2) {
	s << std::setfill('0') << std::setw(places) << n;
}

void printTime(const chr::microseconds& t) {
	std::stringstream stream;
	chr::microseconds us = t;

	stream << "[";
	printFill(stream, convert<chr::minutes>(us));
	stream << ':';
	printFill(stream, convert<chr::seconds>(us));
	stream << '.';
	printFill(stream, convert<chr::milliseconds>(us), 3);
	stream << '.';
	printFill(stream, us.count(), 3);
	stream << "] " << t.count() << " us";

	std::cout << std::setw(ELAPSED_W) << stream.str();
}

void printTime(const std::string& title, const chr::microseconds& elapsed) {
	std::cout << title;
	printTime(elapsed);
	std::cout << std::setw(1) << '\n';
}

float testApprox(
	const std::vector<float>& massQ, const size_t queryCount, const size_t dim, const size_t k,
	hnswlib::HierarchicalNSW<float>& apprAlg, Answers& answers
) {
	size_t correct = 0;
	size_t total = 0;

	for(int i = 0; i < queryCount; i++) {
		std::unordered_set<hnswlib::labeltype> g;
		PriorityQueue gt(answers[i]);
		auto result = apprAlg.searchKnn(&massQ[i * dim], k);

		total += gt.size();

		while(gt.size()) {
			g.insert(gt.top().second);
			gt.pop();
		}

		while(result.size()) {
			if(g.find(result.top().second) != g.end())
				correct++;
			result.pop();
		}
	}

	return 1.f * correct / total;
}

void testVsRecall(
	const std::vector<float>& massQ, const size_t queryCount, const size_t dim, const size_t k,
	hnswlib::HierarchicalNSW<float>& apprAlg, Answers& answers
) {
	std::vector<size_t> efs;

	for(size_t i = 10; i < 30; i++)
		efs.push_back(i);

	for(size_t i = 100; i < 2000; i += 100)
		efs.push_back(i);

	std::cout << std::setw(EF_W) << "EF" << std::setw(RECALL_W) << "Recall" << std::setw(ELAPSED_W) << "Total time" <<
		std::setw(ELAPSED_W) << "Time per query" << std::setw(1) << '\n';

	for(const auto& ef : efs) {
		apprAlg.setEf(ef);
		StopW stopW{};

		const auto recall = testApprox(massQ, queryCount, dim, k, apprAlg, answers);
		const auto elapsed = stopW.elapsedMicro();
		const auto elapsedPerQuery = chr::microseconds(elapsed / queryCount);

		std::cout << std::setw(EF_W) << ef << std::setw(RECALL_W) << recall;
		printTime(elapsed);
		printTime(elapsedPerQuery);
		std::cout << std::setw(1) << '\n';

		if(recall > 1.f)
			throw std::runtime_error("Recall " + std::to_string(recall) + " is greater than 1.");
	}
}

template<typename T>
std::vector<T> read(const fs::path& datasetsDir, const std::string& filename) {
	const auto path = datasetsDir / (filename + ".bin");

	if(!fs::exists(path))
		throw std::runtime_error("Path " + path.string() + " doesn't exist.");

	std::ifstream s(path, std::ios::binary);

	if(!s)
		throw std::runtime_error("Could not open " + path.string());

	s.seekg(0, std::ios::end);
	const auto size = s.tellg();
	s.seekg(0, std::ios::beg);

	std::vector<T> res;
	res.resize(size / sizeof(T));
	s.read(reinterpret_cast<std::ifstream::char_type*>(res.data()), size);
	return res;
}

int main() {
	try {
		const auto datasetsDir = fs::path(SOLUTION_DIR) / "datasets";
		constexpr size_t DIM = 128;
		constexpr size_t K = 10;

		const auto mass = read<float>(datasetsDir, "sift1M");
		const auto massQ = read<float>(datasetsDir, "siftQ1M");
		const auto massQA = read<unsigned int>(datasetsDir, "knnQA1M");
		const auto elemCount = mass.size() / DIM;
		const auto queryCount = massQ.size() / DIM;

		hnswlib::L2Space space(DIM);
		hnswlib::HierarchicalNSW<float> apprAlg(&space, elemCount, 16, 200);

		std::cout << "Building index.\n";

		{
			StopW stopW{};

			for(int i = 0; i < elemCount; i++)
				apprAlg.addPoint(&mass[i * DIM], size_t(i));

			printTime("Index built.", stopW.elapsedMicro());
		}
		
		Answers answers;
		std::cout << "Loading " << (USE_BRUTEFORCE ? "bruteforce" : "results from file") << ".\n";

		{
			StopW stopW{};

			if constexpr(USE_BRUTEFORCE)
				getGt(mass, elemCount, massQ, queryCount, DIM, K, space, answers);
			else
				getGt(mass, elemCount, massQ, queryCount, massQA, DIM, K, space, answers);

			printTime("Loaded.", stopW.elapsedMicro());
		}

		testVsRecall(massQ, queryCount, DIM, K, apprAlg, answers);
		return EXIT_SUCCESS;

	} catch(const std::runtime_error& e) {
		std::cerr << "[ERROR] " << e.what() << '\n';
	}

	return EXIT_FAILURE;
}
