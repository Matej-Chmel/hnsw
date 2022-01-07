#pragma once
#include <Baca/hnsw.h>
#include <hnswlib/hnswalg.h>
#include "KNNAlgo.hpp"

namespace chm {
	class BacaWrapper : public HNSWAlgo {
		size_t ef;
		HNSW* hnsw;

	protected:
		void init() override;
		void insert(float* data, size_t) override;

	public:
		static const std::string NAME;

		~BacaWrapper();
		BacaWrapper(const HNSWConfigPtr& cfg);
		IdxVec3DPtr getConnections() const override;
		KNNResultPtr search(const FloatVecPtr& coords, size_t K) override;
		void setSearchEF(size_t ef) override;
	};

	class hnswlibWrapper : public HNSWAlgo {
		hnswlib::HierarchicalNSW<float>* hnsw;
		hnswlib::L2Space* space;

	protected:
		void init() override;
		void insert(float* data, size_t idx) override;

	public:
		static const std::string NAME;

		~hnswlibWrapper();
		IdxVec3DPtr getConnections() const override;
		hnswlibWrapper(const HNSWConfigPtr& cfg);
		KNNResultPtr search(const FloatVecPtr& coords, size_t K) override;
		void setSearchEF(size_t ef) override;
	};
}
