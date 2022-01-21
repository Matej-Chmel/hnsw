#include "chm/Hnsw.hpp"
#include "runner.hpp"

namespace chm {
	void run() {
		std::vector<float> coords{1.f, 2.f, 3.f, 4.f};

		chm::Hnsw<float, size_t, true> hnsw(2, 200, 16, 2, 100);
		hnsw.insert(coords.data(), 0);
		hnsw.insert(coords.data() + 2, 1);
	}
}
