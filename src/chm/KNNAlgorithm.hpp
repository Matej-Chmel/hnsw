#pragma once
#include <string>
#include "dataOps.hpp"
#include "Unique.hpp"

namespace chm {
	struct KNNResult {
		FloatVec2D distances;
		IdxVec2D indices;

		void resize(size_t queryCount);
	};

	typedef std::shared_ptr<KNNResult> KNNResultPtr;

	class KNNAlgorithm : public Unique {
		std::string info;

	protected:
		KNNAlgorithm(const std::string& info);

	public:
		virtual ~KNNAlgorithm() = default;
		virtual void build(const FloatVecPtr& coords) = 0;
		std::string getInfo() const;
		virtual KNNResultPtr search(const FloatVecPtr& coords, size_t K) = 0;
	};

	class TrueKNNAlgorithm : public KNNAlgorithm {
	protected:
		TrueKNNAlgorithm(const std::string& info);

	public:
		virtual ~TrueKNNAlgorithm() = default;
	};

	struct HNSWConfig {
		const size_t dim;
		const size_t efConstruction;
		const size_t M;
		const size_t maxElements;
		const size_t seed;

		HNSWConfig(size_t dim, size_t efConstruction, size_t M, size_t maxElements, size_t seed);
	};

	typedef std::shared_ptr<HNSWConfig> HNSWConfigPtr;

	class HNSWAlgorithm : public KNNAlgorithm {
	protected:
		const HNSWConfigPtr cfg;

		size_t getElementCount(const FloatVecPtr& coords) const;
		HNSWAlgorithm(const HNSWConfigPtr& cfg, const std::string& info);

	public:
		virtual ~HNSWAlgorithm() = default;
		virtual IdxVec3DPtr getConnections() const = 0;
		virtual void setSearchEF(size_t ef) = 0;
	};
}
