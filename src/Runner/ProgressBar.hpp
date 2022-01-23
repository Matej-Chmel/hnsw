#pragma once
#include <chrono>
#include <iostream>
#include <string>

namespace chm {
	namespace chr = std::chrono;

	using LL = long long;

	class ProgressBar {
		std::string bar;
		float blockTicks;
		float current;
		size_t drawPos;
		float nextUpdate;
		size_t numsIdx;
		size_t numsLen;
		chr::system_clock::time_point start;
		float total;
		float updateTicks;

		LL diffInSeconds();
		void writeNums();

	public:
		ProgressBar(const std::string& title, const size_t total, const size_t width);
		void update();
	};

	void streamTime(std::ostream& s, const LL seconds);
}
