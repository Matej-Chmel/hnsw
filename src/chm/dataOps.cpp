#include <algorithm>
#include <fstream>
#include <random>
#include "AppError.hpp"
#include "dataOps.hpp"
#include "literals.hpp"

namespace chm {
	void throwMissingComponents(size_t actualLen, size_t expectedLen) {
		throw AppError("Not enough components in file.\nAvailable: "_f << actualLen << "\nExpected: " << expectedLen);
	}

	void throwNotOpened(const fs::path& p) {
		throw AppError("File "_f << p << " couldn't be opened.");
	}

	void throwUnsupportedExtension(const fs::path& p) {
		throw AppError("File "_f << p << " has unsupported extension.");
	}

	ElementGenerator::ElementGenerator(size_t count, size_t dim, float min, float max, unsigned int seed)
		: count(count), dim(dim), min(min), max(max), seed(seed) { }

	FloatVecPtr ElementGenerator::generate() {
		std::default_random_engine gen(this->seed);
		std::uniform_real_distribution<float> dist(this->min, this->max);
		auto res = std::make_shared<FloatVec>();

		const size_t len = count * dim;
		res->reserve(len);

		for(size_t i = 0; i < len; i++)
			res->push_back(dist(gen));

		return res;
	}

	void ensureDir(const fs::path& p) {
		if(!fs::exists(p))
			fs::create_directory(p);
	}

	FloatVecPtr read(const fs::path& p, size_t count, size_t dim) {
		auto ext = p.extension().string();

		if (!fs::exists(p))
			throw AppError("File "_f << p << " doesn't exist.");

		std::ifstream s(p, std::ios::binary);

		if (!s.is_open())
			throwNotOpened(p);

		if (ext == ".bin")
			return readBin(s, count, dim);
		else
			throwUnsupportedExtension(p);
	}

	FloatVecPtr readBin(std::istream& s, size_t count, size_t dim) {
		s.seekg(0, std::ios::end);
		auto fileSize = s.tellg();
		s.seekg(0, std::ios::beg);

		auto actualLen = fileSize / sizeof(float);
		auto expectedLen = count * dim;

		if (fileSize % sizeof(float))
			throw AppError("Not all components in file are floats.");
		if (actualLen < expectedLen)
			throwMissingComponents(actualLen, expectedLen);

		auto coords = std::make_shared<FloatVec>();
		coords->resize(expectedLen);
		s.read(reinterpret_cast<std::istream::char_type*>(coords->data()), expectedLen * sizeof(float));
		return coords;
	}

	void sortConnections(IdxVec3DPtr& conn) {
		for(auto& nodeLayers : *conn)
			for(auto& layer : nodeLayers)
				std::sort(layer.begin(), layer.end());
	}

	void write(const FloatVecPtr& coords, const fs::path& p, size_t count, size_t dim) {
		auto ext = p.extension().string();
		std::ofstream s(p, std::ios::binary);

		if (!s.is_open())
			throwNotOpened(p);

		if (ext == ".bin")
			writeBin(coords, s, count, dim);
		else
			throwUnsupportedExtension(p);
	}

	void writeBin(const FloatVecPtr& coords, std::ostream& s, size_t count, size_t dim) {
		auto expectedLen = count * dim;

		if (coords->size() < expectedLen)
			throwMissingComponents(coords->size(), expectedLen);

		s.write(reinterpret_cast<std::ostream::char_type*>(coords->data()), expectedLen * sizeof(float));
	}

	void writeConnections(const IdxVec3DPtr& conn, std::ostream& o) {
		const auto& l = *conn;
		const auto nodeCount = l.size();
		const auto nodeLastIdx = nodeCount - 1;

		for(size_t nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++) {
			o << "Node " << nodeIdx << '\n';

			const auto& nodeLayers = l[nodeIdx];
			const auto nodeLayersLen = nodeLayers.size();

			for(size_t layerIdx = nodeLayersLen - 1;; layerIdx--) {
				o << "Layer " << layerIdx << ": ";

				const auto& layer = nodeLayers[layerIdx];

				if(layer.empty()) {
					o << "EMPTY\n";
					continue;
				}

				const auto lastIdx = layer.size() - 1;

				for(size_t i = 0; i < lastIdx; i++)
					o << layer[i] << ' ';

				o << layer[lastIdx] << '\n';

				if(layerIdx == 0)
					break;
			}

			if(nodeIdx != nodeLastIdx)
				o << '\n';
		}
	}
}
