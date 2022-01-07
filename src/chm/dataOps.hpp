#pragma once
#include <filesystem>
#include <iostream>
#include <memory>
#include <vector>

namespace chm {
	namespace fs = std::filesystem;

	typedef std::vector<float> FloatVec;
	typedef std::vector<FloatVec> FloatVec2D;
	typedef std::shared_ptr<FloatVec> FloatVecPtr;

	typedef std::vector<size_t> IdxVec;
	typedef std::vector<IdxVec> IdxVec2D;
	typedef std::vector<IdxVec2D> IdxVec3D;
	typedef std::shared_ptr<IdxVec> IdxVecPtr;
	typedef std::shared_ptr<IdxVec3D> IdxVec3DPtr;

	struct ElementGen {
		size_t count;
		size_t dim;
		float min;
		float max;
		unsigned int seed;

		ElementGen(size_t count, size_t dim, float min, float max, unsigned int seed);
		FloatVecPtr generate() const;
	};

	typedef std::shared_ptr<ElementGen> ElementGenPtr;

	void ensureDir(const fs::path& p);
	FloatVecPtr read(const fs::path& p, size_t count, size_t dim);
	FloatVecPtr readBin(std::istream& s, size_t count, size_t dim);
	void sortConnections(IdxVec3DPtr& conn);
	void write(const FloatVecPtr& coords, const fs::path& p, size_t count, size_t dim);
	void writeBin(const FloatVecPtr& coords, std::ostream& s, size_t count, size_t dim);
	void writeConnections(const IdxVec3DPtr& conn, std::ostream& stream);
}
