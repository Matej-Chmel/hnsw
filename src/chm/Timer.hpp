#pragma once
#include <chrono>
#include "Unique.hpp"

namespace chm {
	namespace chr = std::chrono;

	class Timer : public Unique {
		chr::system_clock::time_point from;

	public:
		void start();
		long long stop();
		Timer();
	};
}
