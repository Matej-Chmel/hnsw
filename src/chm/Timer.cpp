#include "Timer.hpp"

namespace chm {
	void Timer::start() {
		this->from = chr::system_clock::now();
	}

	long long Timer::stop() {
		auto end = chr::system_clock::now();
		return chr::duration_cast<chr::milliseconds>(end - this->from).count();
	}

	Timer::Timer() {}
}
