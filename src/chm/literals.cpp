#include "literals.hpp"

namespace chm {
	namespace literals {
		std::stringstream operator"" _f(const char* const s, const size_t _) {
			return std::stringstream() << s;
		}
	}
}
