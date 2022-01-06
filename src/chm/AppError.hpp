#pragma once
#include <exception>
#include <sstream>
#include <string>

namespace chm {
	class AppError : public std::exception {
	public:
		AppError(const char* const msg);
		AppError(const std::string& msg);
		AppError(const std::stringstream& s);
	};
}
