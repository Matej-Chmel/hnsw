#pragma once
#include <hnswlib/hnswlib.h>

namespace templatedHnswlib {
	template<typename Coord>
	static Coord innerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);

	template<typename Coord>
	class IPSpace : public hnswlib::SpaceInterface<Coord> {
		hnswlib::DISTFUNC<Coord> fstdistfunc_;
		size_t data_size_;
		size_t dim_;

	public:
		size_t get_data_size() override;
		hnswlib::DISTFUNC<Coord> get_dist_func() override;
		void* get_dist_func_param() override;
		IPSpace(const size_t dim);
	};

	template<typename Coord>
	static Coord euclideanDist(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

	template<typename Coord>
	class EuclideanSpace : public hnswlib::SpaceInterface<Coord> {
		hnswlib::DISTFUNC<Coord> fstdistfunc_;
		size_t data_size_;
		size_t dim_;

	public:
		EuclideanSpace(const size_t dim);
		size_t get_data_size() override;
		hnswlib::DISTFUNC<Coord> get_dist_func() override;
		void* get_dist_func_param() override;
	};

	template<typename Coord>
	Coord innerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr) {
		size_t qty = *((size_t*)qty_ptr);
		Coord res = 0;
		for(unsigned i = 0; i < qty; i++) {
			res += ((Coord*)pVect1)[i] * ((Coord*)pVect2)[i];
		}
		return (1.0f - res);
	}

	template<typename Coord>
	inline size_t IPSpace<Coord>::get_data_size() {
		return this->data_size_;
	}

	template<typename Coord>
	inline hnswlib::DISTFUNC<Coord> IPSpace<Coord>::get_dist_func() {
		return this->fstdistfunc_;
	}

	template<typename Coord>
	inline void* IPSpace<Coord>::get_dist_func_param() {
		return &this->dim_;
	}

	template<typename Coord>
	inline IPSpace<Coord>::IPSpace(const size_t dim) {
		this->fstdistfunc_ = innerProduct;
		this->dim_ = dim;
		this->data_size_ = dim * sizeof(Coord);
	}

	template<typename Coord>
	Coord euclideanDist(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		Coord* pVect1 = (Coord*)pVect1v;
		Coord* pVect2 = (Coord*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);

		Coord res = 0;
		for(size_t i = 0; i < qty; i++) {
			Coord t = *pVect1 - *pVect2;
			pVect1++;
			pVect2++;
			res += t * t;
		}
		return (res);
	}

	template<typename Coord>
	inline EuclideanSpace<Coord>::EuclideanSpace(const size_t dim) {
		this->fstdistfunc_ = euclideanDist;
		this->dim_ = dim;
		this->data_size_ = dim * sizeof(Coord);
	}

	template<typename Coord>
	inline size_t EuclideanSpace<Coord>::get_data_size() {
		return this->data_size_;
	}

	template<typename Coord>
	inline hnswlib::DISTFUNC<Coord> EuclideanSpace<Coord>::get_dist_func() {
		return this->fstdistfunc_;
	}

	template<typename Coord>
	inline void* EuclideanSpace<Coord>::get_dist_func_param() {
		return &this->dim_;
	}
}
