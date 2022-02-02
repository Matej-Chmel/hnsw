#pragma once
#include <chrono>
#include <filesystem>
#include <memory>
#include <vector>

namespace chm {
	namespace chr = std::chrono;
	namespace fs = std::filesystem;

	using LL = long long;

	void padTimeNum(std::ostream& s, const LL num, const size_t places = 2);

	template<typename T>
	using VecPtr = std::shared_ptr<std::vector<T>>;
}
