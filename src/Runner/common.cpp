#include <iomanip>
#include "common.hpp"

namespace chm {
	void padTimeNum(std::ostream& s, const LL num, const size_t places) {
		s << std::setfill('0') << std::setw(places) << num;
	}
}
