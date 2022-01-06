#include "AppError.hpp"

namespace chm {
	AppError::AppError(const char* const msg) : std::exception(msg) {}
	AppError::AppError(const std::string& msg) : AppError(msg.c_str()) {}
	AppError::AppError(const std::stringstream& s) : AppError(s.str()) {}
}
