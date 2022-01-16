#pragma once
#include "Action.hpp"

namespace chm {
	class Runner : public Unique {
		std::vector<ActionPtr> actions;
		CommonState state;
		std::ostream* stream;

		void run(ActionPtr& a);

	public:
		int runAll();
		Runner(const HNSWConfigPtr& cfg, const ElementGenPtr& gen, const std::vector<HNSWAlgoKind>& algoKinds, bool track, std::ostream& stream);
	};
}
