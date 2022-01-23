#pragma once
#include <filesystem>
#include <fstream>
#include <ios>
#include <numeric>
#include "chm/Hnsw.hpp"
#include "ProgressBar.hpp"
#include "templatedHnswlib.hpp"

namespace chm {
	namespace fs = std::filesystem;

	template<typename Coord>
	class HnswlibGateway : public IHnsw<Coord> {
		std::unique_ptr<hnswlib::HierarchicalNSW<Coord>> hnsw;
		std::unique_ptr<hnswlib::SpaceInterface<Coord>> space;

	public:
		IdxVec3DPtr getConnections() const override;
		HnswlibGateway(const HnswCfgPtr& cfg);
		void insert(const ConstIter<Coord>& query) override;
		void knnSearch(
			const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
		) override;
	};

	template<typename Coord>
	using VecPtr = std::shared_ptr<std::vector<Coord>>;

	template<typename Coord>
	class ICoords : public Unique {
		VecPtr<Coord> coords;

		virtual VecPtr<Coord> create() const = 0;

	public:
		virtual ~ICoords() = default;
		ICoords();
		VecPtr<Coord> get();
	};

	template<typename Coord>
	using ICoordsPtr = std::shared_ptr<ICoords<Coord>>;

	template<typename Coord>
	class RndCoords : public ICoords<Coord> {
		size_t count;
		size_t dim;
		Coord min;
		Coord max;
		unsigned int seed;

		VecPtr<Coord> create() const override;

	public:
		RndCoords(const size_t count, const size_t dim, const Coord min, const Coord max, const unsigned int seed);
	};

	template<typename Coord>
	class ReadCoords : public ICoords<Coord> {
		fs::path p;

		VecPtr<Coord> create() const override;

	public:
		ReadCoords(const fs::path& p);
	};

	using LL = long long;

	struct BuildTimeRes : public Unique {
		double avgInsertMS;
		LL initMS;
		std::vector<LL> insertMS;
		LL totalMS;

		BuildTimeRes(const size_t nodeCount);
		void calcStats();
		void print(const std::string& name, std::ostream& s);
	};

	struct RunRes : public Unique {
		bool levelsPassed;
		bool neighborIndicesPassed;
		bool neighborLengthsPassed;
		bool nodeCountPassed;
		BuildTimeRes refBuild;
		BuildTimeRes subBuild;

		void print(std::ostream& s);
		RunRes(const size_t nodeCount);
	};

	using RunResPtr = std::shared_ptr<RunRes>;

	enum class HnswKind {
		CHM_AUTO,
		CHM_INT,
		CHM_SIZE_T,
		CHM_SHORT,
		HNSWLIB
	};

	template<typename Coord>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg, const HnswKind k);

	template<typename Coord, typename Idx>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg);

	struct HnswRunCfg : public Unique {
		const HnswKind refKind;
		const HnswKind subKind;

		HnswRunCfg(const HnswKind refKind, const HnswKind subKind);
	};

	using HnswRunCfgPtr = std::shared_ptr<HnswRunCfg>;

	template<typename Coord>
	class HnswRunner : public Unique {
		const HnswCfgPtr algoCfg;
		const ICoordsPtr<Coord> coords;
		IdxVec3DPtr refConn;
		const HnswRunCfgPtr runCfg;
		IdxVec3DPtr subConn;

		IdxVec3DPtr build(const HnswKind kind, BuildTimeRes& res);
		size_t getNodeCount();
		bool testLevels();
		bool testNeighborLengths();
		bool testNeighborIndices();
		bool testNodeCount();

	public:
		HnswRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg);
		RunResPtr run();
	};

	IdxVec3DPtr sortedInPlace(const IdxVec3DPtr& conn);

	class Timer : public Unique {
		chr::system_clock::time_point from;

	public:
		void start();
		LL stopMS();
		Timer();
	};

	template<typename Coord>
	using UniformDistribution = std::conditional_t<
		std::is_integral<Coord>::value, std::uniform_int_distribution<Coord>,
		std::conditional_t<std::is_floating_point<Coord>::value, std::uniform_real_distribution<Coord>, void>
	>;

	template<typename Coord>
	inline IdxVec3DPtr HnswlibGateway<Coord>::getConnections() const {
		auto res = std::make_shared<IdxVec3D>(this->hnsw->cur_element_count);
		auto& r = *res;

		for(size_t i = 0; i < this->hnsw->cur_element_count; i++) {
			auto& nodeLayers = r[i];
			const auto nodeLayersLen = size_t(this->hnsw->element_levels_[i]) + 1;
			nodeLayers.resize(nodeLayersLen);

			for(size_t level = 0; level < nodeLayersLen; level++) {
				const auto& linkList = this->hnsw->get_linklist_at_level(hnswlib::tableint(i), int(level));
				const auto linkListLen = this->hnsw->getListCount(linkList);
				auto& neighbors = nodeLayers[level];
				neighbors.reserve(linkListLen);

				for(size_t i = 1; i <= linkListLen; i++)
					neighbors.push_back(linkList[i]);
			}
		}

		return res;
	}

	template<typename Coord>
	inline HnswlibGateway<Coord>::HnswlibGateway(const HnswCfgPtr& cfg) {
		if(cfg->useEuclid)
			this->space = std::make_unique<templatedHnswlib::EuclideanSpace<Coord>>(cfg->dim);
		else
			this->space = std::make_unique<templatedHnswlib::IPSpace<Coord>>(cfg->dim);

		this->hnsw = std::make_unique<hnswlib::HierarchicalNSW<Coord>>(this->space.get(), cfg->maxNodeCount, cfg->M, cfg->efConstruction, cfg->seed);
	}

	template<typename Coord>
	inline void HnswlibGateway<Coord>::insert(const ConstIter<Coord>& query) {
		this->hnsw->addPoint(&*query, this->hnsw->cur_element_count);
	}

	template<typename Coord>
	inline void HnswlibGateway<Coord>::knnSearch(
		const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
	) {
		auto results = this->hnsw->searchKnn(&*query, K);
		const auto len = results.size();

		resDistances.clear();
		resDistances.reserve(len);
		resIndices.clear();
		resIndices.reserve(len);

		while(!results.empty()) {
			{
				const auto& item = results.top();
				resDistances.push_back(item.first);
				resIndices.push_back(item.second);
			}
			results.pop();
		}
	}

	template<typename Coord>
	inline ICoords<Coord>::ICoords() : coords(nullptr) {}

	template<typename Coord>
	inline VecPtr<Coord> ICoords<Coord>::get() {
		if(!this->coords)
			this->coords = this->create();
		return this->coords;
	}

	template<typename Coord>
	inline VecPtr<Coord> RndCoords<Coord>::create() const {
		UniformDistribution<Coord> dist(this->min, this->max);
		std::default_random_engine gen(this->seed);

		auto res = std::make_shared<std::vector<Coord>>();
		auto& r = *res;
		const auto total = this->count * this->dim;

		ProgressBar bar("Generating coordinates.", total, 32);
		r.reserve(total);

		for(size_t i = 0; i < total; i++) {
			r.push_back(dist(gen));
			bar.update();
		}

		return res;
	}

	template<typename Coord>
	inline RndCoords<Coord>::RndCoords(const size_t count, const size_t dim, const Coord min, const Coord max, const unsigned int seedd)
		: count(count), dim(dim), min(min), max(max), seed(seed) {}

	template<typename Coord>
	inline VecPtr<Coord> ReadCoords<Coord>::create() const {
		std::ifstream s(this->p);

		s.seekg(0, std::ios::end);
		const auto fileSize = s.tellg();
		s.seekg(0, std::ios::beg);

		auto res = std::make_shared<std::vector<Coord>>();
		res->resize(fileSize / sizeof(Coord));
		s.read(reinterpret_cast<std::ifstream::char_type*>(res->data()), fileSize);
		return res;
	}

	template<typename Coord>
	inline ReadCoords<Coord>::ReadCoords(const fs::path& p) : p(p) {}

	template<typename Coord>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg, const HnswKind k) {
		switch(k) {
			case HnswKind::CHM_AUTO:
				return createHnsw<Coord>(cfg);
			case HnswKind::CHM_INT:
				return createHnsw<Coord, unsigned int>(cfg);
			case HnswKind::CHM_SHORT:
				return createHnsw<Coord, unsigned short>(cfg);
			case HnswKind::CHM_SIZE_T:
				return createHnsw<Coord, size_t>(cfg);
			case HnswKind::HNSWLIB:
				return std::make_shared<HnswlibGateway<Coord>>(cfg);
			default:
				return nullptr;
		}
	}

	template<typename Coord, typename Idx>
	IHnswPtr<Coord> createHnsw(const HnswCfgPtr& cfg) {
		if(cfg->useEuclid)
			return std::make_shared<Hnsw<Coord, Idx, true>>(cfg);
		return std::make_shared<Hnsw<Coord, Idx, false>>(cfg);
	}

	template<typename Coord>
	inline IdxVec3DPtr HnswRunner<Coord>::build(const HnswKind kind, BuildTimeRes& res) {
		Timer timer{};

		timer.start();
		auto hnsw = createHnsw<Coord>(this->algoCfg, kind);
		res.initMS = timer.stopMS();

		const auto& coords = *this->coords->get();
		const auto len = this->getNodeCount();

		ProgressBar bar("Inserting elements.", len, 32);

		for(size_t i = 0; i < len; i++) {
			timer.start();
			hnsw->insert(coords.cbegin() + i * this->algoCfg->dim);
			const auto insTime = timer.stopMS();

			res.insertMS.push_back(insTime);
			bar.update();
		}

		res.calcStats();
		return sortedInPlace(hnsw->getConnections());
	}

	template<typename Coord>
	inline size_t HnswRunner<Coord>::getNodeCount() {
		return this->coords->get()->size() / this->algoCfg->dim;
	}

	template<typename Coord>
	inline bool HnswRunner<Coord>::testLevels() {
		const auto len = std::min(this->refConn->size(), this->subConn->size());

		for(size_t i = 0; i < len; i++)
			if((*this->refConn)[i].size() != (*this->subConn)[i].size())
				return false;

		return true;
	}

	template<typename Coord>
	inline bool HnswRunner<Coord>::testNeighborLengths() {
		const auto len = std::min(this->refConn->size(), this->subConn->size());

		for(size_t i = 0; i < len; i++) {
			const auto& nodeLayersRef = (*this->refConn)[i];
			const auto& nodeLayersSub = (*this->subConn)[i];
			const auto nodeLayersLen = std::min(nodeLayersRef.size(), nodeLayersSub.size());

			for(size_t layerIdx = 0; layerIdx < nodeLayersLen; layerIdx++)
				if(nodeLayersRef[layerIdx].size() != nodeLayersSub[layerIdx].size())
					return false;
		}

		return true;
	}

	template<typename Coord>
	inline bool HnswRunner<Coord>::testNeighborIndices() {
		const auto len = std::min(this->refConn->size(), this->subConn->size());

		for(size_t i = 0; i < len; i++) {
			const auto& nodeLayersRef = (*this->refConn)[i];
			const auto& nodeLayersSub = (*this->subConn)[i];
			const auto nodeLayersLen = std::min(nodeLayersRef.size(), nodeLayersSub.size());

			for(size_t layerIdx = 0; layerIdx < nodeLayersLen; layerIdx++) {
				const auto& neighborsRef = nodeLayersRef[layerIdx];
				const auto& neighborsSub = nodeLayersSub[layerIdx];
				const auto neighborsLen = std::min(neighborsRef.size(), neighborsSub.size());

				for(size_t neighborIdx = 0; neighborIdx < neighborsLen; neighborIdx++)
					if(neighborsRef[neighborIdx] != neighborsSub[neighborIdx])
						return false;
			}
		}

		return true;
	}

	template<typename Coord>
	inline bool HnswRunner<Coord>::testNodeCount() {
		const auto count = this->getNodeCount();
		return this->refConn->size() == count && this->subConn->size() == count;
	}

	template<typename Coord>
	inline HnswRunner<Coord>::HnswRunner(const HnswCfgPtr& algoCfg, const ICoordsPtr<Coord>& coords, const HnswRunCfgPtr& runCfg)
		: algoCfg(algoCfg), coords(coords), runCfg(runCfg) {}

	template<typename Coord>
	inline RunResPtr HnswRunner<Coord>::run() {
		auto res = std::make_shared<RunRes>(this->getNodeCount());
		this->refConn = this->build(this->runCfg->refKind, res->refBuild);
		this->subConn = this->build(this->runCfg->subKind, res->subBuild);
		res->levelsPassed = this->testLevels();
		res->neighborIndicesPassed = this->testNeighborIndices();
		res->neighborLengthsPassed = this->testNeighborLengths();
		res->nodeCountPassed = this->testNodeCount();
		return res;
	}
}
