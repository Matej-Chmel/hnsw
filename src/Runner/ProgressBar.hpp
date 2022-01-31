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
		chr::steady_clock::time_point start;
		float total;
		float updateTicks;

		LL diffInSeconds();
		void writeNums();

	public:
		ProgressBar(const std::string& title, const size_t total, const size_t width);
		void update();
	};

	void padTimeNum(std::ostream& s, const LL num, const size_t places = 2);
}
