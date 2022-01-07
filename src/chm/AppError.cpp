#include "AppError.hpp"

namespace chm {
	AppError::AppError(const char* const msg) : std::exception(msg) {}
	AppError::AppError(const std::string& msg) : AppError(msg.c_str()) {}
	AppError::AppError(const std::stringstream& s) : AppError(s.str()) {}

	namespace literals {
		std::stringstream operator"" _f(const char* const s, const size_t _) {
			return std::stringstream() << s;
		}
	}
}
