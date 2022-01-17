#pragma once

///////////////// Select one from these options
//#define MAIN_CONVERT_HDF_TO_BIN
//#define MAIN_UNIT_TESTING
//#define MAIN_RUN_CREATE_AND_QUERY
#define MAIN_RUN_CREATE_AND_QUERY_WITHOUT_HDF5

//#define DEBUG_NET // TODO implement assert
#define COLLECT_STAT

#define VISIT_HASH // visited_ is implemented using a hash map

//#define COUNT_INWARD_DEGREE
//#define COMPUTE_APPROXIMATE_VECTOR
//#define APR_DEBUG

constexpr int kX = 255;

#ifdef COMPUTE_APPROXIMATE_VECTOR

///////////////// Select one from these options
//#define USE_PLAIN_CHAR
#define USE_TRESHOLD_SUMMARY
//#define USE_TWO_FIXED_MAX


constexpr uint8_t DISTANCE_TRESHOLD = 16;

constexpr int EXPLORE_COUNT_TRESHOLD = 10;

constexpr float VECTOR_FRAGMENT1 = 0.1;
constexpr float VECTOR_FRAGMENT2 = 0.15;
#endif

#define DEBUG_MAKE_PUBLIC
#define DECIDE_BY_INDEX
#define NO_APPR_DISTANCES
#define NO_EXTRA_NEIGHBOR
#define SEARCH_FROM_NEAREST_NEIGHBOR
#define STORE_NODE_ORDER
#define USE_STD_RANDOM
