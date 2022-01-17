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
		Runner(
			const std::vector<HNSWAlgoKind>& algoKinds,
			const HNSWConfigPtr& cfg,
			bool debugBuild,
			const ElementGenPtr& gen,
			std::ostream& stream,
			bool track
		);
	};
}
