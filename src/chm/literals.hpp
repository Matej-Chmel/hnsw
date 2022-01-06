#pragma once
#include <sstream>

namespace chm {
	namespace literals {
		std::stringstream operator"" _f(const char* const s, const size_t _);
	}

	using namespace literals;
}
