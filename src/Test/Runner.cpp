#include <cstdlib>
#include "chm/refImplWrappers.hpp"
#include "Runner.hpp"

namespace chm {
	void Runner::run(ActionPtr& a) {
		*this->stream << a->run(&this->state).msg << '\n';
	}

	int Runner::runAll() {
		try {
			for(auto& a : this->actions)
				this->run(a);
		} catch(AppError& e) {
			std::cerr << "[ERROR] " << e.what() << '\n';
			return EXIT_FAILURE;
		}
		return EXIT_SUCCESS;
	}

	Runner::Runner(const HNSWConfigPtr& cfg, const ElementGenPtr& gen, const std::vector<HNSWAlgoKind>& algoKinds, bool track, std::ostream& stream)
		: state(cfg, gen, algoKinds, fs::path(SOLUTION_DIR) / "logs" / "soleRunner"), stream(&stream) {

		this->actions = {
			std::make_shared<ActionGenElements>(),
			std::make_shared<ActionBuildGraphs>(track)
		};
		const auto len = algoKinds.size();

		for(size_t i = 1; i < len; i++)
			this->actions.push_back(std::make_shared<TestConnections>(algoKinds[0], algoKinds[i]));
	}
}
