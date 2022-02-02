#pragma once
#include <fstream>
#include <ios>
#include "common.hpp"
#include "HnswIntermediate.hpp"
#include "ProgressBar.hpp"

namespace chm {
	template<typename Coord>
	class ICoords : public Unique {
		VecPtr<Coord> coords;

		virtual VecPtr<Coord> create() const = 0;

	public:
		virtual ~ICoords() = default;
		ICoords();
		VecPtr<Coord> get();
	};

	template<typename Coord>
	using ICoordsPtr = std::shared_ptr<ICoords<Coord>>;

	template<typename Coord>
	class ReadCoords : public ICoords<Coord> {
		const size_t count;
		const size_t dim;
		const fs::path p;

		VecPtr<Coord> create() const override;

	public:
		ReadCoords(const fs::path& p, const size_t count, const size_t dim);
	};

	template<typename Coord>
	class RndCoords : public ICoords<Coord> {
		size_t count;
		size_t dim;
		Coord min;
		Coord max;
		unsigned int seed;

		VecPtr<Coord> create() const override;

	public:
		RndCoords(const size_t count, const size_t dim, const Coord min, const Coord max, const unsigned int seed);
	};

	template<typename Coord>
	using UniformDistribution = std::conditional_t<
		std::is_integral<Coord>::value, std::uniform_int_distribution<Coord>,
		std::conditional_t<std::is_floating_point<Coord>::value, std::uniform_real_distribution<Coord>, void>
	>;

	template<typename Coord>
	inline ICoords<Coord>::ICoords() : coords(nullptr) {}

	template<typename Coord>
	inline VecPtr<Coord> ICoords<Coord>::get() {
		if(!this->coords)
			this->coords = this->create();
		return this->coords;
	}

	template<typename Coord>
	inline VecPtr<Coord> ReadCoords<Coord>::create() const {
		const auto len = this->count * this->dim;
		auto res = std::make_shared<std::vector<Coord>>();
		std::ifstream s(this->p, std::ios::binary);

		res->resize(len);
		s.read(reinterpret_cast<std::ifstream::char_type*>(res->data()), len * sizeof(Coord));
		return res;
	}

	template<typename Coord>
	inline ReadCoords<Coord>::ReadCoords(const fs::path& p, const size_t count, const size_t dim) : count(count), dim(dim), p(p) {}

	template<typename Coord>
	inline VecPtr<Coord> RndCoords<Coord>::create() const {
		UniformDistribution<Coord> dist(this->min, this->max);
		std::default_random_engine gen(this->seed);

		auto res = std::make_shared<std::vector<Coord>>();
		auto& r = *res;
		const auto total = this->count * this->dim;

		ProgressBar bar("Generating coordinates.", total, 32);
		r.reserve(total);

		for(size_t i = 0; i < total; i++) {
			r.push_back(dist(gen));
			bar.update();
		}

		return res;
	}

	template<typename Coord>
	inline RndCoords<Coord>::RndCoords(const size_t count, const size_t dim, const Coord min, const Coord max, const unsigned int seed)
		: count(count), dim(dim), min(min), max(max), seed(seed) {}
}
