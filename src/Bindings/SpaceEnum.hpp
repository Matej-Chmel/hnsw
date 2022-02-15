#pragma once
#include "pybind.hpp"

namespace chm {
	enum class SpaceEnum {
		EUCLIDEAN,
		INNER_PRODUCT
	};

	void bindSpaceEnum(py::module_& m);
}
