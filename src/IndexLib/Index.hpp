#pragma once
#include "Runner/HnswRunner.hpp"

namespace chm {
	using IndexBuildResPtr = std::shared_ptr<SeqAlgoBuildRes>;

	template<typename Coord>
	using IndexSearchResPtr = std::shared_ptr<SeqAlgoSearchRes<Coord>>;

	template<typename Coord>
	class Index : public Unique {
		IHnswPtr<Coord> hnsw;
		HnswTypePtr type;

		size_t getDim() const;

	public:
		IndexBuildResPtr build(const ICoordsPtr<Coord>& coords);
		Index(const HnswTypePtr& type);
		IndexSearchResPtr<Coord> search(const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors);
	};

	template<typename Coord>
	using IndexPtr = std::shared_ptr<Index<Coord>>;

	template<typename Coord>
	ICoordsPtr<Coord> getRndCoords(const size_t count, const size_t dim, const Coord min, const Coord max, const unsigned int seed);

	template<typename Coord>
	inline size_t Index<Coord>::getDim() const {
		return this->type->cfg->dim;
	}

	template<typename Coord>
	inline IndexBuildResPtr Index<Coord>::build(const ICoordsPtr<Coord>& coords) {
		const auto count = coords->getCount(this->getDim());
		const auto data = coords->get();
		auto res = std::make_shared<SeqAlgoBuildRes>(count);

		ProgressBar bar("Inserting elements.", count);
		Timer timer{};
		Timer totalTimer{};
		totalTimer.start();

		timer.start();
		this->hnsw = createHnsw<Coord>(type);
		res->init = timer.stop();

		for(size_t i = 0; i < count; i++) {
			timer.start();
			this->hnsw->insert(data->cbegin() + i * this->getDim());
			res->queryTime.queries.push_back(timer.stop());
			bar.update();
		}

		res->total = totalTimer.stop();
		res->queryTime.calcStats();
		return res;
	}

	template<typename Coord>
	inline Index<Coord>::Index(const HnswTypePtr& type) : hnsw(nullptr), type(type) {}

	template<typename Coord>
	inline IndexSearchResPtr<Coord> Index<Coord>::search(const SearchCfgPtr<Coord>& cfg, const FoundNeighborsPtr<Coord>& trueNeighbors) {
		const auto count = cfg->coords->getCount(this->getDim());
		const auto data = cfg->coords->get();
		auto res = std::make_shared<SeqAlgoSearchRes<Coord>>(count);

		ProgressBar bar(progressBarTitleANN(cfg->ef), count);
		Timer timer{};
		Timer totalTimer{};
		totalTimer.start();

		for(size_t i = 0; i < count; i++) {
			timer.start();
			this->hnsw->knnSearch(data->cbegin() + i * this->getDim(), cfg->K, cfg->ef, res->neighbors.indices[i], res->neighbors.distances[i]);
			res->queryTime.queries.push_back(timer.stop());
			bar.update();
		}

		res->total = totalTimer.stop();
		res->calcRecall(trueNeighbors->indices);
		res->queryTime.calcStats();
		return res;
	}

	template<typename Coord>
	ICoordsPtr<Coord> getRndCoords(const size_t count, const size_t dim, const Coord min, const Coord max, const unsigned int seed) {
		return std::make_shared<RndCoords<Coord>>(count, dim, min, max, seed);
	}
}
