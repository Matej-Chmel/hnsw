#pragma once
#include <string>
#include "AppError.hpp"
#include "DebugHNSW.hpp"

namespace chm {
	struct KNNResult {
		FloatVec2D distances;
		IdxVec2D indices;

		void resize(size_t queryCount);
	};

	typedef std::shared_ptr<KNNResult> KNNResultPtr;

	class KNNAlgo : public Unique {
		std::string name;

	protected:
		KNNAlgo(const std::string& name);

	public:
		virtual ~KNNAlgo() = default;
		virtual void build(const FloatVecPtr& coords) = 0;
		std::string getName() const;
		virtual KNNResultPtr search(const FloatVecPtr& coords, size_t K) = 0;
	};

	class TrueKNNAlgo : public KNNAlgo {
	protected:
		TrueKNNAlgo(const std::string& name);

	public:
		virtual ~TrueKNNAlgo() = default;
	};

	struct HNSWConfig {
		const size_t dim;
		const size_t efConstruction;
		const size_t M;
		const size_t maxElements;
		const unsigned int seed;

		HNSWConfig(size_t dim, size_t efConstruction, size_t M, size_t maxElements, unsigned int seed);
	};

	typedef std::shared_ptr<HNSWConfig> HNSWConfigPtr;

	class HNSWAlgo : public KNNAlgo {
	protected:
		const HNSWConfigPtr cfg;

		size_t getElementCount(const FloatVecPtr& coords) const;
		HNSWAlgo(const HNSWConfigPtr& cfg, const std::string& name);
		virtual void insert(float* data, size_t idx) = 0;

	public:
		virtual ~HNSWAlgo() = default;
		void build(const FloatVecPtr& coords) override;
		IdxVec3DPtr buildAndTrack(const FloatVecPtr& coords, std::ostream& stream);
		virtual IdxVec3DPtr getConnections() const = 0;
		virtual DebugHNSW* getDebugObject() = 0;
		virtual void init() = 0;
		virtual void setSearchEF(size_t ef) = 0;
	};

	typedef std::shared_ptr<HNSWAlgo> HNSWAlgoPtr;
}
