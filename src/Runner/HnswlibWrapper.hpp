#pragma once
#include "chm/Hnsw.hpp"

namespace chm {
	template<typename Coord>
	class HnswlibInterImpl;

	template<typename Coord>
	class HnswlibWrapper : public IHnsw<Coord> {
		friend HnswlibInterImpl<Coord>;

		std::unique_ptr<hnswlib::HierarchicalNSW<Coord>> hnsw;
		std::unique_ptr<hnswlib::SpaceInterface<Coord>> space;

	public:
		IdxVec3DPtr getConnections() const override;
		HnswlibWrapper(const HnswCfgPtr& cfg);
		void insert(const ConstIter<Coord>& query) override;
		void knnSearch(
			const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
		) override;
	};

	template<typename Coord>
	inline IdxVec3DPtr HnswlibWrapper<Coord>::getConnections() const {
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
	inline HnswlibWrapper<Coord>::HnswlibWrapper(const HnswCfgPtr& cfg) {
		if(cfg->useEuclid)
			this->space = std::make_unique<templatedHnswlib::EuclideanSpace<Coord>>(cfg->dim);
		else
			this->space = std::make_unique<templatedHnswlib::IPSpace<Coord>>(cfg->dim);

		this->hnsw = std::make_unique<hnswlib::HierarchicalNSW<Coord>>(this->space.get(), cfg->maxNodeCount, cfg->M, cfg->efConstruction, cfg->seed);
	}

	template<typename Coord>
	inline void HnswlibWrapper<Coord>::insert(const ConstIter<Coord>& query) {
		this->hnsw->addPoint(&*query, this->hnsw->cur_element_count);
	}

	template<typename Coord>
	inline void HnswlibWrapper<Coord>::knnSearch(
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
}
