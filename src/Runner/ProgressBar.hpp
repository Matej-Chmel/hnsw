#pragma once
#include <iostream>
#include <string>
#include "common.hpp"

namespace chm {
	class ProgressBar {
		std::string bar;
		float blockTicks;
		float current;
		size_t drawPos;
		float nextUpdate;
		size_t numsIdx;
		size_t numsLen;
		chr::steady_clock::time_point start;
		float total;
		float updateTicks;

		LL diffInSeconds();
		void writeNums();

	public:
		ProgressBar(const std::string& title, const size_t total, const size_t width = 32);
		void update();
	};
}
