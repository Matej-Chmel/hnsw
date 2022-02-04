#pragma once
#include "chm/Hnsw.hpp"
#include "common.hpp"
#include "ProgressBar.hpp"

namespace chm {
	template<typename Coord>
	using FoundNeighborsPtr = std::shared_ptr<FoundNeighbors<Coord>>;

	template<typename Coord, bool useEuclid>
	FoundNeighborsPtr<Coord> bruteforce(const VecPtr<Coord>& nodes, const VecPtr<Coord>& queries, const size_t dim, const size_t K) {
		const auto nodeCount = nodes->size() / dim;
		const auto queryCount = queries->size() / dim;
		auto res = std::make_shared<FoundNeighbors<Coord>>(queryCount);

		ProgressBar bar("Bruteforce searching.", nodeCount * queryCount);

		for(size_t queryIdx = 0; queryIdx < queryCount; queryIdx++) {
			NearHeap<Coord, size_t> heap{};
			const auto query = queries->cbegin() + queryIdx * dim;

			heap.reserve(nodeCount);

			for(size_t nodeIdx = 0; nodeIdx < nodeCount; nodeIdx++) {
				if constexpr(useEuclid)
					heap.push(euclideanDistance<Coord>(nodes->cbegin() + nodeIdx * dim, query, dim), nodeIdx);
				else
					heap.push(innerProductDistance<Coord>(nodes->cbegin() + nodeIdx * dim, query, dim), nodeIdx);
				bar.update();
			}

			const auto len = std::min(heap.len(), K);
			auto& distances = res->distances[queryIdx];
			auto& indices = res->indices[queryIdx];

			distances.reserve(len);
			indices.reserve(len);

			for(size_t i = 0; i < len; i++) {
				{
					auto& n = heap.top();
					distances.push_back(n.dist);
					indices.push_back(n.idx);
				}
				heap.pop();
			}
		}

		return res;
	}

	template<typename Coord>
	FoundNeighborsPtr<Coord> readTrueNeighbors(const fs::path& p, const size_t K, const size_t Kmax) {
		std::ifstream s(p, std::ios::binary);

		s.seekg(0, std::ios::end);
		const auto size = s.tellg();
		s.seekg(0, std::ios::beg);

		const auto len = size / sizeof(unsigned int);
		std::vector<unsigned int> indices;
		const auto queryCount = len / Kmax;
		auto res = std::make_shared<FoundNeighbors<Coord>>(queryCount);

		indices.resize(len);
		s.read(reinterpret_cast<std::ifstream::char_type*>(indices.data()), size);

		for(size_t queryIdx = 0; queryIdx < queryCount; queryIdx++) {
			const auto idx = queryIdx * Kmax;
			auto& queryIndices = res->indices[queryIdx];
			queryIndices.reserve(K);

			for(size_t neighborIdx = 0; neighborIdx < K; neighborIdx++)
				queryIndices.push_back(size_t(indices[idx + neighborIdx]));
		}

		return res;
	}
}
