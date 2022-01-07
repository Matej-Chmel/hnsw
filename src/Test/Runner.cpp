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

	Runner::Runner(const HNSWConfigPtr& cfg, const ElementGenPtr& gen, std::ostream& stream)
		: state(cfg, gen, fs::path(SOLUTION_DIR) / "logs" / "soleRunner"), stream(&stream) {

		this->actions = {
			std::make_shared<ActionGenElements>(),
			std::make_shared<ActionBuildGraphs>(),
			std::make_shared<TestNodeCount>(hnswlibWrapper::NAME, BacaWrapper::NAME),
			std::make_shared<TestLevels>(hnswlibWrapper::NAME, BacaWrapper::NAME),
			std::make_shared<TestNeighborsLength>(hnswlibWrapper::NAME, BacaWrapper::NAME),
			std::make_shared<TestNeighborsIndices>(hnswlibWrapper::NAME, BacaWrapper::NAME)
		};
	}
}
