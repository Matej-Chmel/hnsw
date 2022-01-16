#include "debugWrappers.hpp"

namespace chm {
	void directDebugInsert(DebugHNSW* debugObj, float* data, size_t idx) {
		debugObj->startInsert(data, idx);
		debugObj->prepareUpperSearch();

		auto range = debugObj->getUpperRange();

		if(range.shouldLoop)
			for(auto lc = range.start; lc > range.end; lc--)
				debugObj->searchUpperLayers(lc);

		range = debugObj->getLowerRange();

		if(range.shouldLoop) {
			debugObj->prepareLowerSearch();

			for(auto lc = range.start;; lc--) {
				debugObj->searchLowerLayers(lc);
				debugObj->selectOriginalNeighbors(lc);
				debugObj->connect(lc);
				debugObj->prepareNextLayer(lc);

				if(lc == 0)
					break;
			}
		}

		debugObj->setupEnterPoint();
	}

	HnswlibLocals::~HnswlibLocals() {
		this->clear();
	}

	void HnswlibLocals::clear() {
		this->cur_c = 0;
		this->curdist = 0.f;
		this->curlevel = 0;
		this->currObj = 0;
		this->data_point = nullptr;
		this->enterpoint_copy = 0;
		this->epDeleted = false;
		this->isFirstElement = false;
		this->isUpdate = false;
		this->level = -1;
		this->maxlevelcopy = 0;
		this->Mcurmax = 0;
		this->shouldUpperSearch = false;
		this->top_candidates = PriorityQueue{};
	}

	NodeVecPtr DebugHnswlib::vecFromTopCandidates() {
		PriorityQueue candCopy = this->local.top_candidates;
		auto res = std::make_shared<NodeVec>();
		auto& r = *res;

		r.resize(candCopy.size());

		for(auto i = r.size() - 1;; i--) {
			auto& cand = candCopy.top();
			auto& node = r[i];
			node.distance = cand.first;
			node.idx = cand.second;

			candCopy.pop();

			if(i == 0)
				break;
		}

		return res;
	}

	DebugHnswlib::DebugHnswlib(hnswlib::HierarchicalNSW<float>* hnsw) : hnsw(hnsw), local{} {}

	void DebugHnswlib::startInsert(float* coords, size_t label) {
		// Reset local variables.
		this->local.clear();
		this->local.data_point = coords;

		this->local.cur_c = 0;
		{
			// Checking if the element with the same label already exists
			// if so, updating it *instead* of creating a new element.
			// std::unique_lock <std::mutex> templock_curr(this->hnsw->cur_element_count_guard_);
			auto search = this->hnsw->label_lookup_.find(label);
			if(search != this->hnsw->label_lookup_.end()) {
				hnswlib::tableint existingInternalId = search->second;
				// templock_curr.unlock();

				// std::unique_lock <std::mutex> lock_el_update(this->hnsw->link_list_update_locks_[(existingInternalId & (hnswlib::HierarchicalNSW<float>::max_update_element_locks - 1))]);

				if(this->hnsw->isMarkedDeleted(existingInternalId)) {
					this->hnsw->unmarkDeletedInternal(existingInternalId);
				}
				this->hnsw->updatePoint(this->local.data_point, existingInternalId, 1.0);

				// return existingInternalId;
			}

			if(this->hnsw->cur_element_count >= this->hnsw->max_elements_) {
				throw std::runtime_error("The number of elements exceeds the specified limit");
			};

			this->local.cur_c = this->hnsw->cur_element_count;
			this->hnsw->cur_element_count++;
			this->hnsw->label_lookup_[label] = this->local.cur_c;
		}

		// Take update lock to prevent race conditions on an element with insertion/update at the same time.
		// std::unique_lock <std::mutex> lock_el_update(link_list_update_locks_[(cur_c & (max_update_element_locks - 1))]);
		// std::unique_lock <std::mutex> lock_el(link_list_locks_[cur_c]);
		this->local.curlevel = this->hnsw->getRandomLevel(this->hnsw->mult_);
		if(this->local.level > 0)
			this->local.curlevel = this->local.level;

		this->hnsw->element_levels_[this->local.cur_c] = this->local.curlevel;

		// std::unique_lock <std::mutex> templock(global);
		this->local.maxlevelcopy = this->hnsw->maxlevel_;
		/*
		if (curlevel <= maxlevelcopy)
			templock.unlock();
		*/
		this->local.currObj = this->hnsw->enterpoint_node_;
		this->local.enterpoint_copy = this->hnsw->enterpoint_node_;

		memset(this->hnsw->data_level0_memory_ + this->local.cur_c * this->hnsw->size_data_per_element_ + this->hnsw->offsetLevel0_, 0, this->hnsw->size_data_per_element_);

		// Initialisation of the data and label
		memcpy(this->hnsw->getExternalLabeLp(this->local.cur_c), &label, sizeof(hnswlib::labeltype));
		memcpy(this->hnsw->getDataByInternalId(this->local.cur_c), this->local.data_point, this->hnsw->data_size_);

		if(this->local.curlevel) {
			this->hnsw->linkLists_[this->local.cur_c] = (char*)malloc(this->hnsw->size_links_per_element_ * this->local.curlevel + 1);
			if(this->hnsw->linkLists_[this->local.cur_c] == nullptr)
				throw std::runtime_error("Not enough memory: addPoint failed to allocate linklist");
			memset(this->hnsw->linkLists_[this->local.cur_c], 0, this->hnsw->size_links_per_element_ * this->local.curlevel + 1);
		}
	}

	size_t DebugHnswlib::getLatestLevel() {
		return size_t(this->local.curlevel);
	}

	void DebugHnswlib::prepareUpperSearch() {
		this->local.isFirstElement = (signed)this->local.currObj == -1;
		this->local.shouldUpperSearch = !this->local.isFirstElement && this->local.curlevel < this->local.maxlevelcopy;

		if(this->local.shouldUpperSearch)
			this->local.curdist = this->hnsw->fstdistfunc_(this->local.data_point, this->hnsw->getDataByInternalId(this->local.currObj), this->hnsw->dist_func_param_);
	}

	LevelRange DebugHnswlib::getUpperRange() {
		if(this->local.maxlevelcopy < 0)
			return {0, 0, false};

		return {size_t(this->local.maxlevelcopy), size_t(this->local.curlevel), true};
	}

	void DebugHnswlib::searchUpperLayers(size_t level) {
		if(this->local.shouldUpperSearch) {
			bool changed = true;
			while(changed) {
				changed = false;
				unsigned int* data;
				// std::unique_lock <std::mutex> lock(link_list_locks_[currObj]);
				data = this->hnsw->get_linklist(this->local.currObj, level);
				int size = this->hnsw->getListCount(data);

				hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
				for(int i = 0; i < size; i++) {
					hnswlib::tableint cand = datal[i];
					if(cand < 0 || cand > this->hnsw->max_elements_)
						throw std::runtime_error("cand error");
					float d = this->hnsw->fstdistfunc_(this->local.data_point, this->hnsw->getDataByInternalId(cand), this->hnsw->dist_func_param_);
					if(d < this->local.curdist) {
						this->local.curdist = d;
						this->local.currObj = cand;
						changed = true;
					}
				}
			}
		}
	}

	Node DebugHnswlib::getNearestNode() {
		return {this->local.curdist, this->local.currObj};
	}

	void DebugHnswlib::prepareLowerSearch() {
		this->local.epDeleted = this->hnsw->isMarkedDeleted(this->local.enterpoint_copy);
	}

	LevelRange DebugHnswlib::getLowerRange() {
		if(this->local.maxlevelcopy < 0)
			return {0, 0, false};

		return {size_t(std::min(this->local.curlevel, this->local.maxlevelcopy)), 0, true};
	}

	void DebugHnswlib::searchLowerLayers(size_t level) {
		if(level > this->local.maxlevelcopy || level < 0)  // possible?
			throw std::runtime_error("Level error");

		this->local.top_candidates = this->hnsw->searchBaseLayer(this->local.currObj, this->local.data_point, level);
	}

	NodeVecPtr DebugHnswlib::getLowerLayerResults() {
		return this->vecFromTopCandidates();
	}

	void DebugHnswlib::selectOriginalNeighbors(size_t level) {
		if(this->local.epDeleted) {
			this->local.top_candidates.emplace(this->hnsw->fstdistfunc_(this->local.data_point, this->hnsw->getDataByInternalId(this->local.enterpoint_copy), this->hnsw->dist_func_param_), this->local.enterpoint_copy);
			if(this->local.top_candidates.size() > this->hnsw->ef_construction_)
				this->local.top_candidates.pop();
		}

		// currObj = mutuallyConnectNewElement(data_point, cur_c, top_candidates, level, false);
		this->local.isUpdate = false;

		// Start of method mutuallyConnectNewElement.
		this->local.Mcurmax = level ? this->hnsw->maxM_ : this->hnsw->maxM0_;
		this->hnsw->getNeighborsByHeuristic2(this->local.top_candidates, this->hnsw->M_);
	}

	NodeVecPtr DebugHnswlib::getOriginalNeighbors(size_t level) {
		return this->vecFromTopCandidates();
	}

	void DebugHnswlib::connect(size_t level) {
		if(this->local.top_candidates.size() > this->hnsw->M_)
			throw std::runtime_error("Should be not be more than M_ candidates returned by the heuristic");

		std::vector<hnswlib::tableint> selectedNeighbors;
		selectedNeighbors.reserve(this->hnsw->M_);
		while(this->local.top_candidates.size() > 0) {
			selectedNeighbors.push_back(this->local.top_candidates.top().second);
			this->local.top_candidates.pop();
		}

		hnswlib::tableint next_closest_entry_point = selectedNeighbors.back();

		{
			hnswlib::linklistsizeint* ll_cur;
			if(level == 0)
				ll_cur = this->hnsw->get_linklist0(this->local.cur_c);
			else
				ll_cur = this->hnsw->get_linklist(this->local.cur_c, level);

			if(*ll_cur && !this->local.isUpdate) {
				throw std::runtime_error("The newly inserted element should have blank link list");
			}
			this->hnsw->setListCount(ll_cur, selectedNeighbors.size());
			hnswlib::tableint* data = (hnswlib::tableint*)(ll_cur + 1);
			for(size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
				if(data[idx] && !this->local.isUpdate)
					throw std::runtime_error("Possible memory corruption");
				if(level > this->hnsw->element_levels_[selectedNeighbors[idx]])
					throw std::runtime_error("Trying to make a link on a non-existent level");

				data[idx] = selectedNeighbors[idx];
			}
		}

		for(size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
			// std::unique_lock <std::mutex> lock(this->hnsw->link_list_locks_[selectedNeighbors[idx]]);

			hnswlib::linklistsizeint* ll_other;
			if(level == 0)
				ll_other = this->hnsw->get_linklist0(selectedNeighbors[idx]);
			else
				ll_other = this->hnsw->get_linklist(selectedNeighbors[idx], level);

			size_t sz_link_list_other = this->hnsw->getListCount(ll_other);

			if(sz_link_list_other > this->local.Mcurmax)
				throw std::runtime_error("Bad value of sz_link_list_other");
			if(selectedNeighbors[idx] == this->local.cur_c)
				throw std::runtime_error("Trying to connect an element to itself");
			if(level > this->hnsw->element_levels_[selectedNeighbors[idx]])
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

			// If cur_c is already present in the neighboring connections of `selectedNeighbors[idx]` then no need to modify any connections or run the heuristics.
			if(!is_cur_c_present) {
				if(sz_link_list_other < this->local.Mcurmax) {
					data[sz_link_list_other] = this->local.cur_c;
					this->hnsw->setListCount(ll_other, sz_link_list_other + 1);
				} else {
					// finding the "weakest" element to replace it with the new one
					float d_max = this->hnsw->fstdistfunc_(this->hnsw->getDataByInternalId(this->local.cur_c), this->hnsw->getDataByInternalId(selectedNeighbors[idx]),
						this->hnsw->dist_func_param_);
					// Heuristic:
					PriorityQueue candidates;
					candidates.emplace(d_max, this->local.cur_c);

					for(size_t j = 0; j < sz_link_list_other; j++) {
						candidates.emplace(
							this->hnsw->fstdistfunc_(this->hnsw->getDataByInternalId(data[j]), this->hnsw->getDataByInternalId(selectedNeighbors[idx]),
								this->hnsw->dist_func_param_), data[j]);
					}

					this->hnsw->getNeighborsByHeuristic2(candidates, this->local.Mcurmax);

					int indx = 0;
					while(candidates.size() > 0) {
						data[indx] = candidates.top().second;
						candidates.pop();
						indx++;
					}

					this->hnsw->setListCount(ll_other, indx);
					// Nearest K:
					/*int indx = -1;
					for (int j = 0; j < sz_link_list_other; j++) {
						dist_t d = fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(rez[idx]), dist_func_param_);
						if (d > d_max) {
							indx = j;
							d_max = d;
						}
					}
					if (indx >= 0) {
						data[indx] = cur_c;
					} */
				}
			}
		}

		// return next_closest_entry_point;
		this->local.currObj = next_closest_entry_point;
	}

	IdxVecPtr DebugHnswlib::getNeighborsForNode(size_t nodeIdx, size_t lc) {
		auto res = std::make_shared<IdxVec>();

		const auto& linkList = this->hnsw->get_linklist_at_level(hnswlib::tableint(nodeIdx), int(lc));
		const auto linkListLen = this->hnsw->getListCount(linkList);
		res->reserve(linkListLen);

		for(size_t i = 1; i <= linkListLen; i++)
			res->push_back(linkList[i]);

		std::sort(res->begin(), res->end());
		return res;
	}

	void DebugHnswlib::prepareNextLayer(size_t lc) {
		/*
		* ep <- W
		* Nothing here because ep is stored in this->local.currObj.
		*/
	}

	void DebugHnswlib::setupEnterPoint() {
		if(this->local.isFirstElement) {
			// Do nothing for the first element
			this->hnsw->enterpoint_node_ = 0;
			this->hnsw->maxlevel_ = this->local.curlevel;
		}

		if(this->local.curlevel > this->local.maxlevelcopy) {
			this->hnsw->enterpoint_node_ = this->local.cur_c;
			this->hnsw->maxlevel_ = this->local.curlevel;
		}
	}

	size_t DebugHnswlib::getEnterPoint() {
		return this->hnsw->enterpoint_node_;
	}

	void hnswlibDebugWrapper::init() {
		hnswlibWrapper::init();
		this->debugObj = new DebugHnswlib(this->hnsw);
	}

	void hnswlibDebugWrapper::insert(float* data, size_t idx) {
		directDebugInsert(this->debugObj, data, idx);
	}

	hnswlibDebugWrapper::~hnswlibDebugWrapper() {
		delete this->debugObj;
	}

	hnswlibDebugWrapper::hnswlibDebugWrapper(const HNSWConfigPtr& cfg) : hnswlibWrapper(cfg, "hnswlib-HNSW-Debug"), debugObj(nullptr) {}
}
