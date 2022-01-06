#pragma once

namespace chm {
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
