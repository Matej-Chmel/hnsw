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

	namespace literals {
		std::stringstream operator"" _f(const char* const s, const size_t _);
	}

	using namespace literals;

	class Unique {
	protected:
		Unique() = default;

	public:
		virtual ~Unique() = default;
		Unique& operator=(const Unique&) = delete;
		Unique& operator=(Unique&&) = delete;
		Unique(const Unique&) = delete;
		Unique(Unique&&) = delete;
	};
}
