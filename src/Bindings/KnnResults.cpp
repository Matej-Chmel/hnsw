#include <stdexcept>
#include "KnnResults.hpp"

namespace chm {
	void bindSpaceEnum(py::module_& m) {
		py::enum_<SpaceEnum>(m, "Space")
			.value("EUCLIDEAN", SpaceEnum::EUCLIDEAN)
			.value("INNER_PRODUCT", SpaceEnum::INNER_PRODUCT)
			.export_values();
	}

	void checkBufInfo(const py::buffer_info& buf, const size_t dim) {
		if (buf.ndim == 2) {
			if (buf.shape[1] != dim)
				throw std::runtime_error(WRONG_FEATURES);
		}
		else if (buf.ndim != 1)
			throw std::runtime_error(WRONG_DIM);
	}

	void freeWhenDone(void* d) {
		delete[] d;
	}
}
