#pragma once
#include "BigNode.hpp"
#include "hnswlibTemplateSpaces.hpp"
#include "HnswlibWrapper.hpp"

namespace chm {
	template<typename Coord>
	using PriorityQueue = std::priority_queue<
		std::pair<Coord, hnswlib::tableint>,
		std::vector<std::pair<Coord, hnswlib::tableint>>,
		typename hnswlib::HierarchicalNSW<Coord>::CompareByFirst
	>;

	template<typename Coord>
	struct IHnswIntermediate : public IHnsw<Coord> {
		virtual ~IHnswIntermediate() = default;
		void insert(const ConstIter<Coord>& query) override;

		virtual void startInsert(const ConstIter<Coord>& query) = 0;
		virtual size_t getLatestLevel() const = 0;
		virtual void prepareUpperSearch() = 0;
		virtual LevelRngPtr getUpperRange() const = 0;
		virtual void searchUpperLayer(const size_t lc) = 0;
		virtual BigNode<Coord> getNearestNode() const = 0;
		virtual void prepareLowerSearch() = 0;
		virtual LevelRngPtr getLowerRange() const = 0;
		virtual size_t getLowerSearchEntry() const = 0;
		virtual void searchLowerLayer(const size_t lc) = 0;
		virtual BigNodeVecPtr<Coord> getLowerLayerResults() const = 0;
		virtual void selectOriginalNeighbors(const size_t lc) = 0;
		virtual BigNodeVecPtr<Coord> getOriginalNeighbors() const = 0;
		virtual void connect(const size_t lc) = 0;
		virtual VecPtr<size_t> getNeighborsForNode(const size_t nodeIdx, const size_t lc) const = 0;
		virtual void setupEnterPoint() = 0;
		virtual size_t getEnterPoint() const = 0;

		virtual void startSearchANN(const ConstIter<Coord>& query, const size_t ef, const size_t K) = 0;
		virtual LevelRngPtr getRangeANN() const = 0;
		virtual void searchUpperLayerANN(const size_t lc) = 0;
		virtual void searchLastLayerANN() = 0;
		virtual BigNodeVecPtr<Coord> getLastLayerResultsANN() const = 0;
		virtual void fillResultsANN(std::vector<size_t>& resIndices, std::vector<Coord>& resDistances) = 0;
	};

	template<typename Coord>
	using IHnswIntermediatePtr = std::shared_ptr<IHnswIntermediate<Coord>>;

	template<typename Coord, typename Idx>
	struct HnswLocals {
		FarHeap<Coord, Idx> candidates;
		size_t ef;
		Node<Coord, Idx> ep;
		size_t K;
		size_t L;
		size_t l;
		size_t layerMmax;
		bool isFirstNode;
		ConstIter<Coord> query;
		FarHeap<Coord, Idx> W;
	};

	template<typename Coord, typename Idx, bool useEuclid>
	class HnswInterImpl : public IHnswIntermediate<Coord> {
		std::unique_ptr<Hnsw<Coord, Idx, useEuclid>> hnsw;
		HnswLocals<Coord, Idx> local;

	public:
		IdxVec3DPtr getConnections() const override;
		HnswInterImpl(const HnswCfgPtr& cfg, const HnswSettingsPtr& settings);
		void knnSearch(
			const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
		) override;

		void startInsert(const ConstIter<Coord>& query) override;
		size_t getLatestLevel() const override;
		void prepareUpperSearch() override;
		LevelRngPtr getUpperRange() const override;
		void searchUpperLayer(const size_t lc) override;
		BigNode<Coord> getNearestNode() const override;
		void prepareLowerSearch() override;
		LevelRngPtr getLowerRange() const override;
		size_t getLowerSearchEntry() const override;
		void searchLowerLayer(const size_t lc) override;
		BigNodeVecPtr<Coord> getLowerLayerResults() const override;
		void selectOriginalNeighbors(const size_t lc) override;
		BigNodeVecPtr<Coord> getOriginalNeighbors() const override;
		void connect(const size_t lc) override;
		VecPtr<size_t> getNeighborsForNode(const size_t nodeIdx, const size_t lc) const override;
		void setupEnterPoint() override;
		size_t getEnterPoint() const override;

		void startSearchANN(const ConstIter<Coord>& query, const size_t ef, const size_t K) override;
		LevelRngPtr getRangeANN() const override;
		void searchUpperLayerANN(const size_t lc) override;
		void searchLastLayerANN() override;
		BigNodeVecPtr<Coord> getLastLayerResultsANN() const override;
		void fillResultsANN(std::vector<size_t>& resIndices, std::vector<Coord>& resDistances) override;
	};

	template<typename Coord>
	struct HnswlibLocals {
		hnswlib::tableint cur_c;
		Coord curdist;
		int curlevel;
		hnswlib::tableint currObj;
		const void* data_point;
		hnswlib::tableint enterpoint_copy;
		bool epDeleted;
		bool isFirstElement;
		bool isUpdate;
		size_t k;
		int level;
		int maxlevelcopy;
		size_t Mcurmax;
		const void* query_data;
		bool shouldUpperSearch;
		PriorityQueue<Coord> top_candidates;
	};

	template<typename Coord>
	class HnswlibInterImpl : public IHnswIntermediate<Coord> {
		HnswlibLocals<Coord> local;
		std::unique_ptr<HnswlibWrapper<Coord>> w;

		BigNodeVecPtr<Coord> vecFromTopCandidates() const;

	public:
		IdxVec3DPtr getConnections() const override;
		HnswlibInterImpl(const HnswCfgPtr& cfg);
		void knnSearch(
			const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
		) override;

		void startInsert(const ConstIter<Coord>& query) override;
		size_t getLatestLevel() const override;
		void prepareUpperSearch() override;
		LevelRngPtr getUpperRange() const override;
		void searchUpperLayer(const size_t lc) override;
		BigNode<Coord> getNearestNode() const override;
		void prepareLowerSearch() override;
		LevelRngPtr getLowerRange() const override;
		size_t getLowerSearchEntry() const override;
		void searchLowerLayer(const size_t lc) override;
		BigNodeVecPtr<Coord> getLowerLayerResults() const override;
		void selectOriginalNeighbors(const size_t lc) override;
		BigNodeVecPtr<Coord> getOriginalNeighbors() const override;
		void connect(const size_t lc) override;
		VecPtr<size_t> getNeighborsForNode(const size_t nodeIdx, const size_t lc) const override;
		void setupEnterPoint() override;
		size_t getEnterPoint() const override;

		void startSearchANN(const ConstIter<Coord>& query, const size_t ef, const size_t K) override;
		LevelRngPtr getRangeANN() const override;
		void searchUpperLayerANN(const size_t lc) override;
		void searchLastLayerANN() override;
		BigNodeVecPtr<Coord> getLastLayerResultsANN() const override;
		void fillResultsANN(std::vector<size_t>& resIndices, std::vector<Coord>& resDistances) override;
	};

	template<typename Coord>
	inline void IHnswIntermediate<Coord>::insert(const ConstIter<Coord>& query) {
		this->startInsert(query);
		this->prepareUpperSearch();

		{
			const auto range = this->getUpperRange();

			if(range)
				for(auto lc = range->start; lc > range->end; lc--)
					this->searchUpperLayer(lc);
		}

		const auto range = this->getLowerRange();

		if(range) {
			this->prepareLowerSearch();

			for(auto lc = range->start;; lc--) {
				this->searchLowerLayer(lc);
				this->selectOriginalNeighbors(lc);
				this->connect(lc);

				if(!lc)
					break;
			}
		}

		this->setupEnterPoint();
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline IdxVec3DPtr HnswInterImpl<Coord, Idx, useEuclid>::getConnections() const {
		return this->hnsw->getConnections();
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline HnswInterImpl<Coord, Idx, useEuclid>::HnswInterImpl(const HnswCfgPtr& cfg, const HnswSettingsPtr& settings)
		: hnsw(std::make_unique<Hnsw<Coord, Idx, useEuclid>>(cfg, settings)), local{} {}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::knnSearch(
		const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
	) {
		this->hnsw->knnSearch(query, K, ef, resIndices, resDistances);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::startInsert(const ConstIter<Coord>& query) {
		std::copy(query, query + this->hnsw->dim, this->hnsw->coords.begin() + this->hnsw->nodeCount * this->hnsw->dim);

		this->local.query = query;
		this->local.isFirstNode = !this->hnsw->nodeCount;

		if(this->local.isFirstNode) {
			this->hnsw->entryLevel = this->hnsw->getNewLevel();
			this->hnsw->nodeCount = 1;
			this->hnsw->connections->init(this->hnsw->entryIdx, this->hnsw->entryLevel);
			return;
		}

		if(this->hnsw->distanceCacheEnabled)
			this->hnsw->distanceCache.clear();

		this->local.ep = Node(
			this->hnsw->getDistance(this->hnsw->getCoords(this->hnsw->entryIdx), query, true, this->hnsw->entryIdx),
			this->hnsw->entryIdx
		);
		this->local.L = this->hnsw->entryLevel;
		this->local.l = this->hnsw->getNewLevel();

		this->hnsw->connections->init(this->hnsw->nodeCount, this->local.l);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline size_t HnswInterImpl<Coord, Idx, useEuclid>::getLatestLevel() const {
		return this->local.l;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::prepareUpperSearch() {
		// Nothing to prepare.
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline LevelRngPtr HnswInterImpl<Coord, Idx, useEuclid>::getUpperRange() const {
		if(this->local.isFirstNode)
			return nullptr;
		return std::make_shared<LevelRng>(this->local.L, this->local.l);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::searchUpperLayer(const size_t lc) {
		this->hnsw->searchUpperLayer(this->local.query, this->local.ep, lc);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline BigNode<Coord> HnswInterImpl<Coord, Idx, useEuclid>::getNearestNode() const {
		return BigNode<Coord>(this->local.ep.dist, size_t(this->local.ep.idx));
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::prepareLowerSearch() {
		// Nothing to prepare.
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline LevelRngPtr HnswInterImpl<Coord, Idx, useEuclid>::getLowerRange() const {
		if(this->local.isFirstNode)
			return nullptr;
		return std::make_shared<LevelRng>(std::min(this->local.L, this->local.l), 0);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline size_t HnswInterImpl<Coord, Idx, useEuclid>::getLowerSearchEntry() const {
		return size_t(this->local.ep.idx);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::searchLowerLayer(const size_t lc) {
		this->local.candidates.clear();
		this->hnsw->searchLowerLayer(this->local.query, this->local.ep, this->hnsw->efConstruction, lc, this->local.candidates);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline BigNodeVecPtr<Coord> HnswInterImpl<Coord, Idx, useEuclid>::getLowerLayerResults() const {
		return copyToVec(this->local.candidates);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::selectOriginalNeighbors(const size_t lc) {
		this->hnsw->selectNeighborsHeuristic(this->local.candidates, this->hnsw->M);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline BigNodeVecPtr<Coord> HnswInterImpl<Coord, Idx, useEuclid>::getOriginalNeighbors() const {
		return copyToVec(this->local.candidates);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::connect(const size_t lc) {
		this->hnsw->connections->useNeighbors(this->hnsw->nodeCount, lc);
		this->hnsw->connections->fillFrom(this->local.candidates);

		// ep = nearest from candidates
		this->local.ep = Node(this->local.candidates.top());
		const auto layerMmax = !lc ? this->hnsw->Mmax0 : this->hnsw->M;

		for(const auto& eIdx : *this->hnsw->connections) {
			this->hnsw->connections->useNeighbors(eIdx, lc);

			if(this->hnsw->connections->len() < layerMmax)
				this->hnsw->connections->push(this->hnsw->nodeCount);
			else {
				this->hnsw->fillHeap(this->hnsw->getCoords(eIdx), this->local.query, this->local.candidates);
				this->hnsw->selectNeighborsHeuristic(this->local.candidates, layerMmax);
				this->hnsw->connections->fillFrom(this->local.candidates);
			}
		}
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline VecPtr<size_t> HnswInterImpl<Coord, Idx, useEuclid>::getNeighborsForNode(const size_t nodeIdx, const size_t lc) const {
		this->hnsw->connections->useNeighbors(Idx(nodeIdx), Idx(lc));
		auto res = std::make_shared<std::vector<size_t>>();
		auto& r = *res;

		r.reserve(this->hnsw->connections->len());

		for(const auto& neighborIdx : *this->hnsw->connections)
			r.push_back(size_t(neighborIdx));

		return res;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::setupEnterPoint() {
		if(this->local.isFirstNode)
			return;

		if(this->local.l > this->local.L) {
			this->hnsw->entryIdx = this->hnsw->nodeCount;
			this->hnsw->entryLevel = this->local.l;
		}

		this->hnsw->nodeCount++;
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline size_t HnswInterImpl<Coord, Idx, useEuclid>::getEnterPoint() const {
		return size_t(this->hnsw->entryIdx);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::startSearchANN(const ConstIter<Coord>& query, const size_t ef, const size_t K) {
		this->local.ef = ef;
		this->local.K = K;
		this->local.query = query;

		if(this->hnsw->distanceCacheEnabled)
			this->hnsw->distanceCache.clear();

		this->local.ep = Node(
			this->hnsw->getDistance(this->hnsw->getCoords(this->hnsw->entryIdx), this->local.query, true, this->hnsw->entryIdx),
			this->hnsw->entryIdx
		);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline LevelRngPtr HnswInterImpl<Coord, Idx, useEuclid>::getRangeANN() const {
		return std::make_shared<LevelRng>(this->hnsw->entryLevel, 0);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::searchUpperLayerANN(const size_t lc) {
		this->hnsw->searchUpperLayer(this->local.query, this->local.ep, lc);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::searchLastLayerANN() {
		this->local.W.clear();
		this->hnsw->searchLowerLayer(this->local.query, this->local.ep, Idx(this->local.ef), 0, this->local.W);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline BigNodeVecPtr<Coord> HnswInterImpl<Coord, Idx, useEuclid>::getLastLayerResultsANN() const {
		return copyToVec(this->local.W);
	}

	template<typename Coord, typename Idx, bool useEuclid>
	inline void HnswInterImpl<Coord, Idx, useEuclid>::fillResultsANN(std::vector<size_t>& resIndices, std::vector<Coord>& resDistances) {
		while(this->local.W.len() > this->local.K)
			this->local.W.pop();

		const auto len = this->local.W.len();
		resDistances.clear();
		resDistances.resize(len);
		resIndices.clear();
		resIndices.resize(len);

		for(auto i = len - 1;; i--) {
			{
				const auto& n = this->local.W.top();
				resDistances[i] = n.dist;
				resIndices[i] = size_t(n.idx);
			}
			this->local.W.pop();

			if(!i)
				break;
		}
	}

	template<typename Coord>
	inline BigNodeVecPtr<Coord> HnswlibInterImpl<Coord>::vecFromTopCandidates() const {
		PriorityQueue<Coord> candCopy = this->local.top_candidates;
		auto res = std::make_shared<BigNodeVec<Coord>>();
		auto& r = *res;

		r.resize(candCopy.size());

		for(auto i = r.size() - 1;; i--) {
			{
				auto& cand = candCopy.top();
				auto& node = r[i];
				node.dist = cand.first;
				node.idx = cand.second;
			}

			candCopy.pop();

			if(!i)
				break;
		}

		return res;
	}

	template<typename Coord>
	inline IdxVec3DPtr HnswlibInterImpl<Coord>::getConnections() const {
		return this->w->getConnections();
	}

	template<typename Coord>
	inline HnswlibInterImpl<Coord>::HnswlibInterImpl(const HnswCfgPtr& cfg)
		: local{}, w(std::make_unique<HnswlibWrapper<Coord>>(cfg)) {}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::knnSearch(
		const ConstIter<Coord>& query, const size_t K, const size_t ef, std::vector<size_t>& resIndices, std::vector<Coord>& resDistances
	) {
		this->w->knnSearch(query, K, ef, resIndices, resDistances);
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::startInsert(const ConstIter<Coord>& query) {
		const auto label = this->w->hnsw->cur_element_count;
		this->local.level = -1;
		this->local.data_point = &*query;

		this->local.cur_c = 0;
		{
			/*
			Checking if the element with the same label already exists
			if so, updating it *instead* of creating a new element.
			std::unique_lock <std::mutex> templock_curr(this->w->hnsw->cur_element_count_guard_);
			*/
			auto search = this->w->hnsw->label_lookup_.find(label);
			if(search != this->w->hnsw->label_lookup_.end()) {
				hnswlib::tableint existingInternalId = search->second;

				/*
				templock_curr.unlock();
				std::unique_lock <std::mutex> lock_el_update(
					this->w->hnsw->link_list_update_locks_[(existingInternalId & (hnswlib::HierarchicalNSW<float>::max_update_element_locks - 1))]
				);
				*/

				if(this->w->hnsw->isMarkedDeleted(existingInternalId)) {
					this->w->hnsw->unmarkDeletedInternal(existingInternalId);
				}
				this->w->hnsw->updatePoint(this->local.data_point, existingInternalId, 1.0);

				// return existingInternalId;
			}

			if(this->w->hnsw->cur_element_count >= this->w->hnsw->max_elements_) {
				throw std::runtime_error("The number of elements exceeds the specified limit");
			};

			this->local.cur_c = this->w->hnsw->cur_element_count;
			this->w->hnsw->cur_element_count++;
			this->w->hnsw->label_lookup_[label] = this->local.cur_c;
		}

		/*
		Take update lock to prevent race conditions on an element with insertion/update at the same time.
		std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
		std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
		*/
		this->local.curlevel = this->w->hnsw->getRandomLevel(this->w->hnsw->mult_);
		if(this->local.level > 0)
			this->local.curlevel = this->local.level;

		this->w->hnsw->element_levels_[this->local.cur_c] = this->local.curlevel;

		// std::unique_lock <std::mutex> templock(global);
		this->local.maxlevelcopy = this->w->hnsw->maxlevel_;
		/*
		if (curlevel <= maxlevelcopy)
			templock.unlock();
		*/
		this->local.currObj = this->w->hnsw->enterpoint_node_;
		this->local.enterpoint_copy = this->w->hnsw->enterpoint_node_;

		memset(
			this->w->hnsw->data_level0_memory_ + this->local.cur_c * this->w->hnsw->size_data_per_element_ + this->w->hnsw->offsetLevel0_,
			0, this->w->hnsw->size_data_per_element_
		);

		// Initialisation of the data and label
		memcpy(this->w->hnsw->getExternalLabeLp(this->local.cur_c), &label, sizeof(hnswlib::labeltype));
		memcpy(this->w->hnsw->getDataByInternalId(this->local.cur_c), this->local.data_point, this->w->hnsw->data_size_);

		if(this->local.curlevel) {
			this->w->hnsw->linkLists_[this->local.cur_c] = (char*)malloc(this->w->hnsw->size_links_per_element_ * this->local.curlevel + 1);
			if(this->w->hnsw->linkLists_[this->local.cur_c] == nullptr)
				throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
			memset(this->w->hnsw->linkLists_[this->local.cur_c], 0, this->w->hnsw->size_links_per_element_ * this->local.curlevel + 1);
		}
	}

	template<typename Coord>
	inline size_t HnswlibInterImpl<Coord>::getLatestLevel() const {
		return this->local.curlevel;
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::prepareUpperSearch() {
		this->local.isFirstElement = (signed)this->local.currObj == -1;
		this->local.shouldUpperSearch = !this->local.isFirstElement && this->local.curlevel < this->local.maxlevelcopy;

		if(this->local.shouldUpperSearch)
			this->local.curdist = this->w->hnsw->fstdistfunc_(
				this->local.data_point, this->w->hnsw->getDataByInternalId(this->local.currObj), this->w->hnsw->dist_func_param_
			);
	}

	template<typename Coord>
	inline LevelRngPtr HnswlibInterImpl<Coord>::getUpperRange() const {
		if(this->local.maxlevelcopy < 0)
			return nullptr;
		return std::make_shared<LevelRng>(size_t(this->local.maxlevelcopy), size_t(this->local.curlevel));
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::searchUpperLayer(const size_t level) {
		if(this->local.shouldUpperSearch) {
			bool changed = true;
			while(changed) {
				changed = false;
				unsigned int* data;
				// std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
				data = this->w->hnsw->get_linklist(this->local.currObj, level);
				int size = this->w->hnsw->getListCount(data);

				hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
				for(int i = 0; i < size; i++) {
					hnswlib::tableint cand = datal[i];
					if(cand < 0 || cand > this->w->hnsw->max_elements_)
						throw std::runtime_error("cand error");
					const auto d = this->w->hnsw->fstdistfunc_(
						this->local.data_point, this->w->hnsw->getDataByInternalId(cand), this->w->hnsw->dist_func_param_
					);
					if(d < this->local.curdist) {
						this->local.curdist = d;
						this->local.currObj = cand;
						changed = true;
					}
				}
			}
		}
	}

	template<typename Coord>
	inline BigNode<Coord> HnswlibInterImpl<Coord>::getNearestNode() const {
		return Node(this->local.curdist, size_t(this->local.currObj));
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::prepareLowerSearch() {
		this->local.epDeleted = this->w->hnsw->isMarkedDeleted(this->local.enterpoint_copy);
	}

	template<typename Coord>
	inline LevelRngPtr HnswlibInterImpl<Coord>::getLowerRange() const {
		if(this->local.maxlevelcopy < 0)
			return nullptr;
		return std::make_shared<LevelRng>(std::min(this->local.curlevel, this->local.maxlevelcopy), 0);
	}

	template<typename Coord>
	inline size_t HnswlibInterImpl<Coord>::getLowerSearchEntry() const {
		return size_t(this->local.currObj);
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::searchLowerLayer(const size_t level) {
		if(level > this->local.maxlevelcopy || level < 0)  // possible?
			throw std::runtime_error("Level error");

		this->local.top_candidates = this->w->hnsw->searchBaseLayer(this->local.currObj, this->local.data_point, level);
	}

	template<typename Coord>
	inline BigNodeVecPtr<Coord> HnswlibInterImpl<Coord>::getLowerLayerResults() const {
		return this->vecFromTopCandidates();
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::selectOriginalNeighbors(const size_t level) {
		if(this->local.epDeleted) {
			this->local.top_candidates.emplace(
				this->w->hnsw->fstdistfunc_(
					this->local.data_point, this->w->hnsw->getDataByInternalId(this->local.enterpoint_copy),
					this->w->hnsw->dist_func_param_
				), this->local.enterpoint_copy
			);

			if(this->local.top_candidates.size() > this->w->hnsw->ef_construction_)
				this->local.top_candidates.pop();
		}

		// currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
		this->local.isUpdate = false;

		// Start of method mutuallyConnectNewElement.
		this->local.Mcurmax = level ? this->w->hnsw->maxM_ : this->w->hnsw->maxM0_;
		this->w->hnsw->getNeighborsByHeuristic2(this->local.top_candidates, this->w->hnsw->M_);
	}

	template<typename Coord>
	inline BigNodeVecPtr<Coord> HnswlibInterImpl<Coord>::getOriginalNeighbors() const {
		return this->vecFromTopCandidates();
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::connect(const size_t level) {
		if(this->local.top_candidates.size() > this->w->hnsw->M_)
			throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

		std::vector<hnswlib::tableint> selectedNeighbors;
		selectedNeighbors.reserve(this->w->hnsw->M_);
		while(this->local.top_candidates.size() > 0) {
			selectedNeighbors.push_back(this->local.top_candidates.top().second);
			this->local.top_candidates.pop();
		}

		hnswlib::tableint next_closest_entry_point = selectedNeighbors.back();

		{
			hnswlib::linklistsizeint* ll_cur;
			if(level == 0)
				ll_cur = this->w->hnsw->get_linklist0(this->local.cur_c);
			else
				ll_cur = this->w->hnsw->get_linklist(this->local.cur_c, level);

			if(*ll_cur && !this->local.isUpdate) {
				throw std::runtime_error("The newly inserted element should have blank link list");
			}
			this->w->hnsw->setListCount(ll_cur, selectedNeighbors.size());
			hnswlib::tableint* data = (hnswlib::tableint*)(ll_cur + 1);
			for(size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
				if(data[idx] && !this->local.isUpdate)
					throw std::runtime_error("Possible memory corruption");
				if(level > this->w->hnsw->element_levels_[selectedNeighbors[idx]])
					throw std::runtime_error("Trying to make a link on a non-existent level");

				data[idx] = selectedNeighbors[idx];
			}
		}

		for(size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
			// std::unique_lock <std::mutex> lock(this->w->hnsw->link_list_locks_[selectedNeighbors[idx]]);

			hnswlib::linklistsizeint* ll_other;
			if(level == 0)
				ll_other = this->w->hnsw->get_linklist0(selectedNeighbors[idx]);
			else
				ll_other = this->w->hnsw->get_linklist(selectedNeighbors[idx], level);

			size_t sz_link_list_other = this->w->hnsw->getListCount(ll_other);

			if(sz_link_list_other > this->local.Mcurmax)
				throw std::runtime_error("Bad value of sz_link_list_other");
			if(selectedNeighbors[idx] == this->local.cur_c)
				throw std::runtime_error("Trying to connect an element to itself");
			if(level > this->w->hnsw->element_levels_[selectedNeighbors[idx]])
				throw std::runtime_error("Trying to make a link on a non-existent level");

			hnswlib::tableint* data = (hnswlib::tableint*)(ll_other + 1);

			bool is_cur_c_present = false;
			if(this->local.isUpdate) {
				for(size_t j = 0; j < sz_link_list_other; j++) {
					if(data[j] == this->local.cur_c) {
						is_cur_c_present = true;
						break;
					}
				}
			}

			/*
			If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]`
			then no need to modify any connections or run the heuristics.
			*/
			if(!is_cur_c_present) {
				if(sz_link_list_other < this->local.Mcurmax) {
					data[sz_link_list_other] = this->local.cur_c;
					this->w->hnsw->setListCount(ll_other, sz_link_list_other + 1);
				} else {
					// Finding the "weakest" element to replace it with the new one
					float d_max = this->w->hnsw->fstdistfunc_(
						this->w->hnsw->getDataByInternalId(this->local.cur_c), this->w->hnsw->getDataByInternalId(selectedNeighbors[idx]),
						this->w->hnsw->dist_func_param_
					);
					// Heuristic:
					PriorityQueue<Coord> candidates;
					candidates.emplace(d_max, this->local.cur_c);

					for(size_t j = 0; j < sz_link_list_other; j++) {
						candidates.emplace(
							this->w->hnsw->fstdistfunc_(
								this->w->hnsw->getDataByInternalId(data[j]), this->w->hnsw->getDataByInternalId(selectedNeighbors[idx]),
								this->w->hnsw->dist_func_param_
							), data[j]
						);
					}

					this->w->hnsw->getNeighborsByHeuristic2(candidates, this->local.Mcurmax);

					int indx = 0;
					while(candidates.size() > 0) {
						data[indx] = candidates.top().second;
						candidates.pop();
						indx++;
					}

					this->w->hnsw->setListCount(ll_other, indx);
					/*
					Nearest K:
					int indx = -1;
					for (int j = 0; j < sz_link_list_other; j++) {
						dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
						if (d > d_max) {
							indx = j;
							d_max = d;
						}
					}
					if (indx >= 0) {
						data[indx] = cur_c;
					}
					*/
				}
			}
		}

		// return next_closest_entry_point;
		this->local.currObj = next_closest_entry_point;
	}

	template<typename Coord>
	inline VecPtr<size_t> HnswlibInterImpl<Coord>::getNeighborsForNode(const size_t nodeIdx, const size_t lc) const {
		auto res = std::make_shared<std::vector<size_t>>();

		const auto& linkList = this->w->hnsw->get_linklist_at_level(hnswlib::tableint(nodeIdx), int(lc));
		const auto linkListLen = this->w->hnsw->getListCount(linkList);
		res->reserve(linkListLen);

		for(size_t i = 1; i <= linkListLen; i++)
			res->push_back(size_t(linkList[i]));

		return res;
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::setupEnterPoint() {
		if(this->local.isFirstElement) {
			// Do nothing for the first element
			this->w->hnsw->enterpoint_node_ = 0;
			this->w->hnsw->maxlevel_ = this->local.curlevel;
			return;
		}

		if(this->local.curlevel > this->local.maxlevelcopy) {
			this->w->hnsw->enterpoint_node_ = this->local.cur_c;
			this->w->hnsw->maxlevel_ = this->local.curlevel;
		}
	}

	template<typename Coord>
	inline size_t HnswlibInterImpl<Coord>::getEnterPoint() const {
		return size_t(this->w->hnsw->enterpoint_node_);
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::startSearchANN(const ConstIter<Coord>& query, const size_t ef, const size_t K) {
		this->local.k = K;
		this->local.query_data = &*query;
		this->w->hnsw->setEf(ef);

		this->local.currObj = this->w->hnsw->enterpoint_node_;
		this->local.curdist = this->w->hnsw->fstdistfunc_(
			this->local.query_data, this->w->hnsw->getDataByInternalId(this->w->hnsw->enterpoint_node_), this->w->hnsw->dist_func_param_
		);
	}

	template<typename Coord>
	inline LevelRngPtr HnswlibInterImpl<Coord>::getRangeANN() const {
		return std::make_shared<LevelRng>(size_t(this->w->hnsw->maxlevel_), 0);
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::searchUpperLayerANN(const size_t level) {
		bool changed = true;
		while(changed) {
			changed = false;
			unsigned int* data;

			data = (unsigned int*)this->w->hnsw->get_linklist(this->local.currObj, level);
			int size = this->w->hnsw->getListCount(data);
			this->w->hnsw->metric_hops++;
			this->w->hnsw->metric_distance_computations += size;

			hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
			for(int i = 0; i < size; i++) {
				hnswlib::tableint cand = datal[i];
				if(cand < 0 || cand > this->w->hnsw->max_elements_)
					throw std::runtime_error("cand error");
				Coord d = this->w->hnsw->fstdistfunc_(
					this->local.query_data, this->w->hnsw->getDataByInternalId(cand), this->w->hnsw->dist_func_param_
				);

				if(d < this->local.curdist) {
					this->local.curdist = d;
					this->local.currObj = cand;
					changed = true;
				}
			}
		}
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::searchLastLayerANN() {
		if(this->w->hnsw->num_deleted_) {
			this->local.top_candidates = this->w->hnsw->searchBaseLayerST<true, true>(
				this->local.currObj, this->local.query_data, std::max(this->w->hnsw->ef_, this->local.k)
			);
		} else {
			this->local.top_candidates = this->w->hnsw->searchBaseLayerST<false, true>(
				this->local.currObj, this->local.query_data, std::max(this->w->hnsw->ef_, this->local.k)
			);
		}
	}

	template<typename Coord>
	inline BigNodeVecPtr<Coord> HnswlibInterImpl<Coord>::getLastLayerResultsANN() const {
		return this->vecFromTopCandidates();
	}

	template<typename Coord>
	inline void HnswlibInterImpl<Coord>::fillResultsANN(std::vector<size_t>& resIndices, std::vector<Coord>& resDistances) {
		std::priority_queue<std::pair<Coord, hnswlib::labeltype>> result;

		while(this->local.top_candidates.size() > this->local.k) {
			this->local.top_candidates.pop();
		}
		while(this->local.top_candidates.size() > 0) {
			std::pair<Coord, hnswlib::tableint> rez = this->local.top_candidates.top();
			result.push(std::pair<Coord, hnswlib::labeltype>(rez.first, this->w->hnsw->getExternalLabel(rez.second)));
			this->local.top_candidates.pop();
		}

		const auto len = result.size();

		resDistances.clear();
		resDistances.reserve(len);
		resIndices.clear();
		resIndices.reserve(len);

		while(!result.empty()) {
			{
				const auto& item = result.top();
				resDistances.push_back(item.first);
				resIndices.push_back(item.second);
			}
			result.pop();
		}
	}
}
