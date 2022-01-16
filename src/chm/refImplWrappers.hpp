#pragma once
#include <Baca/hnsw.h>
#include <hnswlib/hnswalg.h>
#include "KNNAlgo.hpp"

namespace chm {
	class BacaWrapper : public HNSWAlgo {
	protected:
		size_t ef;
		baca::HNSW* hnsw;

		void insert(float* data, size_t) override;

	public:
		virtual ~BacaWrapper();
		BacaWrapper(const HNSWConfigPtr& cfg);
		BacaWrapper(const HNSWConfigPtr& cfg, const std::string& name);
		IdxVec3DPtr getConnections() const override;
		DebugHNSW* getDebugObject() override;
		void init() override;
		KNNResultPtr search(const FloatVecPtr& coords, size_t K) override;
		void setSearchEF(size_t ef) override;
	};

	class hnswlibWrapper : public HNSWAlgo {
	protected:
		hnswlib::HierarchicalNSW<float>* hnsw;
		hnswlib::L2Space* space;

		void insert(float* data, size_t idx) override;

	public:
		virtual ~hnswlibWrapper();
		IdxVec3DPtr getConnections() const override;
		DebugHNSW* getDebugObject() override;
		hnswlibWrapper(const HNSWConfigPtr& cfg);
		hnswlibWrapper(const HNSWConfigPtr& cfg, const std::string& name);
		void init() override;
		KNNResultPtr search(const FloatVecPtr& coords, size_t K) override;
		void setSearchEF(size_t ef) override;
	};
}
