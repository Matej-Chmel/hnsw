#include <stdexcept>
#include "InterErr.hpp"

namespace chm {
	std::string str(const InterTest t) {
		switch(t) {
			case InterTest::ENTRY:
				return "Graph entry point";
			case InterTest::FINAL_NEIGHBORS:
				return "Neighbors after connecting";
			case InterTest::LEVEL:
				return "Last generated level";
			case InterTest::LOWER_SEARCH_ANN:
				return "Result of lower search when running ANN search";
			case InterTest::LOWER_SEARCH_ENTRY:
				return "Entry point of lower search";
			case InterTest::LOWER_SEARCH_RES:
				return "Results of lower search";
			case InterTest::SELECTED_NEIGHBORS:
				return "Original selected neighbors";
			case InterTest::UPPER_SEARCH:
				return "Nearest node from upper search";
			case InterTest::UPPER_SEARCH_ANN:
				return "Nearest node from upper search when running ANN search";
			default:
				throw std::runtime_error("Unknown InterTest value.");
		}
	}
}
