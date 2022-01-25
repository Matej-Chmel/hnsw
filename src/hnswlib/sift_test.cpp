#define DECIDE_BY_IDX
#include <chrono>
#include <filesystem>
#include <hnswlib/hnswlib.h>

namespace chr = std::chrono;
namespace fs = std::filesystem;

class StopW {
	chr::steady_clock::time_point time_begin;

public:
	StopW() {
		this->time_begin = chr::steady_clock::now();
	}

	float getElapsedTimeMicro() const {
		const auto time_end = std::chrono::steady_clock::now();
		return chr::duration_cast<chr::microseconds>(time_end - this->time_begin).count();
	}

	void reset() {
		this->time_begin = chr::steady_clock::now();
	}
};

using PriorityQueue = std::priority_queue<std::pair<float, hnswlib::labeltype>>;
using Answers = std::vector<PriorityQueue>;

void get_gt(
	const float* const mass, const float* const massQ, const size_t vecsize, const size_t qsize, hnswlib::L2Space& l2space,
	const size_t vecdim, Answers& answers, const size_t k
) {
	hnswlib::BruteforceSearch<float> bs(&l2space, vecsize);

	for(int i = 0; i < vecsize; i++)
		bs.addPoint((void*)(mass + vecdim * i), size_t(i));

	(Answers(qsize)).swap(answers);

	for(int i = 0; i < qsize; i++) {
		auto gt = bs.searchKnn(massQ + vecdim * i, 10);
		answers[i] = gt;
	}
}

void get_gt(
	const unsigned int* const massQA, const float* const massQ, const float* const mass, const size_t vecsize, const size_t qsize,
	hnswlib::L2Space& l2space, const size_t vecdim, Answers& answers, const size_t k
) {
	(Answers(qsize)).swap(answers);

	auto fstdistfunc_ = l2space.get_dist_func();

	std::cout << qsize << '\n';

	for(int i = 0; i < qsize; i++) {
		for(int j = 0; j < k; j++) {
			auto other = fstdistfunc_(massQ + i * vecdim, mass + massQA[100 * i + j] * vecdim, l2space.get_dist_func_param());
			answers[i].emplace(other, massQA[100 * i + j]);
		}
	}
}

float test_approx(
	const float* const massQ, const size_t vecsize, const size_t qsize, hnswlib::HierarchicalNSW<float>& appr_alg, const size_t vecdim,
	Answers& answers, const size_t k
) {
	size_t correct = 0;
	size_t total = 0;

	for(int i = 0; i < qsize; i++) {
		auto result = appr_alg.searchKnn(massQ + vecdim * i, 10);
		PriorityQueue gt(answers[i]);
		std::unordered_set<hnswlib::labeltype> g;

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

	return 1.0f * correct / total;
}

int test_vs_recall(
	const float* const massQ, const size_t vecsize, const size_t qsize, hnswlib::HierarchicalNSW<float>& appr_alg, const size_t vecdim,
	Answers& answers, const size_t k
) {
	std::vector<size_t> efs;

	for(int i = 10; i < 30; i++)
		efs.push_back(i);

	for(int i = 100; i < 2000; i += 100)
		efs.push_back(i);

	for(const auto& ef : efs) {
		appr_alg.setEf(ef);
		StopW stopw{};

		auto recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k);
		auto time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

		std::cout << ef << '\t' << recall << '\t' << time_us_per_query << " us\n";

		if(recall > 1.f) {
			std::cout << recall << '\t' << time_us_per_query << " us\n";
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}

int main() {
	const auto datasetsDir = fs::path(SOLUTION_DIR) / "datasets";
	const size_t qsize = 10000;
	const size_t vecdim = 128;
	const size_t vecsize = 1000000;

	auto mass = new float[vecsize * vecdim];
	auto massQ = new float[qsize * vecdim];
	auto massQA = new unsigned int[qsize * 100];

	{
		std::ifstream s(datasetsDir / "sift1M.bin", std::ios::binary);
		s.read((char*)mass, vecsize * vecdim * sizeof(float));
	}
	{
		std::ifstream s(datasetsDir / "siftQ1M.bin", std::ios::binary);
		s.read((char*)massQ, qsize * vecdim * sizeof(float));
	}
	{
		std::ifstream s(datasetsDir / "knnQA1M.bin", std::ios::binary);
		s.read((char*)massQA, qsize * 100 * sizeof(int));
	}

	hnswlib::L2Space l2space(vecdim);
	hnswlib::HierarchicalNSW<float> appr_alg(&l2space, vecsize, 16, 200);

	std::cout << "Building index\n";

	StopW stopwb{};

	for(int i = 0; i < 1; i++)
		appr_alg.addPoint((void*)(mass + vecdim * i), size_t(i));

	for(int i = 1; i < vecsize; i++)
		appr_alg.addPoint((void*)(mass + vecdim * i), size_t(i));

	std::cout << "Index built, time=" << stopwb.getElapsedTimeMicro() * 1e-6 << '\n';

	Answers answers;
	const size_t k = 10;

	std::cout << "Loading gt\n";

	get_gt(massQA, massQ, mass, vecsize, qsize, l2space, vecdim, answers, k);

	std::cout << "Loaded gt\n";

	for(int i = 0; i < 1; i++)
		if(test_vs_recall(massQ, vecsize, qsize, appr_alg, vecdim, answers, k) != EXIT_SUCCESS)
			return EXIT_FAILURE;

	return EXIT_SUCCESS;
}
