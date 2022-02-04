#include "HnswRunner.hpp"

namespace chm {
	std::string progressBarTitleANN(const size_t ef) {
		return "Searching queries, EF = " + std::to_string(ef) + '.';
	}
}
