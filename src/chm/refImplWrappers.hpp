#pragma once
#include <Baca/hnsw.h>
#include <hnswlib/hnswalg.h>
#include "KNNAlgorithm.hpp"

namespace chm {
	class hnswlibWrapper : public HNSWAlgorithm {
		hnswlib::HierarchicalNSW<float>* hnsw;
		hnswlib::L2Space* space;

	public:
		~hnswlibWrapper();
		void build(const FloatVecPtr& coords) override;
		IdxVec3DPtr getConnections() const override;
		hnswlibWrapper(const HNSWConfigPtr& cfg);
		KNNResultPtr search(const FloatVecPtr& coords, size_t K) override;
		void setSearchEF(size_t ef) override;
	};

	class BacaWrapper : public HNSWAlgorithm {
		size_t ef;
		HNSW* hnsw;

	public:
		~BacaWrapper();
		BacaWrapper(const HNSWConfigPtr& cfg);
		void build(const FloatVecPtr& coords) override;
		IdxVec3DPtr getConnections() const override;
		KNNResultPtr search(const FloatVecPtr& coords, size_t K) override;
		void setSearchEF(size_t ef) override;
	};
}
