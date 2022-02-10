#include <algorithm>
#include <sstream>
#include "ProgressBar.hpp"

namespace chm {
	void streamTime(std::ostream& s, const LL seconds) {
		padTimeNum(s, seconds / 60LL);
		s << ':';
		padTimeNum(s, seconds % 60LL);
	}

	LL ProgressBar::diffInSeconds() {
		return chr::duration_cast<chr::seconds>(chr::steady_clock::now() - this->start).count();
	}

	void ProgressBar::writeNums() {
		const auto seconds = this->diffInSeconds();

		std::stringstream s;
		s << size_t(this->current) << " / " << size_t(this->total) << "    " << size_t(this->current * 100.f / this->total) <<
			" %    ";
		streamTime(s, seconds);
		const auto res = s.str();

		if(res.size() > this->numsLen) {
			const auto diff = res.size() - this->numsLen;
			this->numsLen += diff;
			this->bar.resize(this->bar.size() + diff);
		}

		std::copy(res.cbegin(), res.cend(), this->bar.begin() + this->numsIdx);
	}

	ProgressBar::ProgressBar(const std::string& title, const size_t total, const size_t width)
		: blockTicks(float(total) / float(width)), current(0.f), drawPos(2), nextUpdate(this->blockTicks / 2.f),
		numsIdx(width + 7), updateTicks(nextUpdate), total(float(total)) {

		std::cout << title << '\n';

		std::stringstream s;
		s << "\r[" << std::string(width, '_') << "]    0 / 0    0 %    00:00";
		this->bar = s.str();
		this->numsLen = this->bar.size() - this->numsIdx;
		this->start = chr::steady_clock::now();
	}

	void ProgressBar::update() {
		if(this->current >= this->total)
			return;

		this->current++;

		if(this->current < this->nextUpdate)
			return;

		this->nextUpdate += this->updateTicks;

		const auto drawCount = size_t(this->current / this->blockTicks) - (this->drawPos - 2);

		if(drawCount > 0)
			for(size_t i = 0; i < drawCount; i++)
				this->bar[this->drawPos++] = 'O';

		this->writeNums();
		std::cout << this->bar;

		if(this->current == this->total) {
			const auto seconds = this->diffInSeconds();

			std::cout << "\nCompleted in ";
			streamTime(std::cout, seconds);
			std::cout << ".\n\n";
		}
	}
}
