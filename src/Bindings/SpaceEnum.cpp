#include "SpaceEnum.hpp"

namespace chm {
	void bindSpaceEnum(py::module_& m) {
		py::enum_<SpaceEnum>(m, "Space")
			.value("EUCLIDEAN", SpaceEnum::EUCLIDEAN)
			.value("INNER_PRODUCT", SpaceEnum::INNER_PRODUCT)
			.export_values();
	}
}
