#pragma once
#include <Baca/hnsw.h>
#include <hnswlib/hnswalg.h>
#include "KNNAlgo.hpp"

namespace chm {
	class BacaWrapper : public HNSWAlgo {
	protected:
		size_t ef;
		HNSW* hnsw;

		void init() override;
		void insert(float* data, size_t) override;

	public:
		~BacaWrapper();
		BacaWrapper(const HNSWConfigPtr& cfg);
		IdxVec3DPtr getConnections() const override;
		KNNResultPtr search(const FloatVecPtr& coords, size_t K) override;
		void setSearchEF(size_t ef) override;
	};

	class hnswlibWrapper : public HNSWAlgo {
	protected:
		hnswlib::HierarchicalNSW<float>* hnsw;
		hnswlib::L2Space* space;

		void init() override;
		void insert(float* data, size_t idx) override;

	public:
		virtual ~hnswlibWrapper();
		IdxVec3DPtr getConnections() const override;
		hnswlibWrapper(const HNSWConfigPtr& cfg);
		hnswlibWrapper(const HNSWConfigPtr& cfg, const std::string& name);
		KNNResultPtr search(const FloatVecPtr& coords, size_t K) override;
		void setSearchEF(size_t ef) override;
	};
}
