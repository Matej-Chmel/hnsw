#include "chm/chmHnsw.hpp"
#include "runner.hpp"

namespace chm {
	void run() {
		std::vector<float> coords{1.f, 2.f, 3.f, 4.f};

		const auto cfg = std::make_shared<HnswConfig>(2, 200, 16, 2, 100, true);
		auto hnsw = createHnsw<float>(cfg);
		hnsw->insert(coords.cbegin());
		hnsw->insert(coords.cbegin() + 2);

		const auto connections = hnsw->getConnections();
		IdxVec indices;
		std::vector<float> distances;
		hnsw->knnSearch(coords.cbegin(), 2, 2, indices, distances);
		hnsw->knnSearch(coords.cbegin() + 2, 2, 2, indices, distances);
	}
}
