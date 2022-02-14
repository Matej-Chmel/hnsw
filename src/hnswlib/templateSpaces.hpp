// This file was created by Matej Chmel.

#pragma once
#include "hnswlib.h"

namespace templatedHnswlib {
	template<typename dist_t>
	static dist_t innerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr);

	template<typename dist_t>
	class IPSpace : public hnswlib::SpaceInterface<dist_t> {
		hnswlib::DISTFUNC<dist_t> fstdistfunc_;
		size_t data_size_;
		size_t dim_;

	public:
		size_t get_data_size() override;
		hnswlib::DISTFUNC<dist_t> get_dist_func() override;
		void* get_dist_func_param() override;
		IPSpace(const size_t dim);
	};

	template<typename dist_t>
	static dist_t euclideanDist(const void* pVect1v, const void* pVect2v, const void* qty_ptr);

	template<typename dist_t>
	class EuclideanSpace : public hnswlib::SpaceInterface<dist_t> {
		hnswlib::DISTFUNC<dist_t> fstdistfunc_;
		size_t data_size_;
		size_t dim_;

	public:
		EuclideanSpace(const size_t dim);
		size_t get_data_size() override;
		hnswlib::DISTFUNC<dist_t> get_dist_func() override;
		void* get_dist_func_param() override;
	};

	template<typename dist_t>
	dist_t innerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr) {
		size_t qty = *((size_t*)qty_ptr);
		dist_t res = 0;
		for(unsigned i = 0; i < qty; i++) {
			res += ((dist_t*)pVect1)[i] * ((dist_t*)pVect2)[i];
		}
		return (1.0f - res);
	}

	template<typename dist_t>
	inline size_t IPSpace<dist_t>::get_data_size() {
		return this->data_size_;
	}

	template<typename dist_t>
	inline hnswlib::DISTFUNC<dist_t> IPSpace<dist_t>::get_dist_func() {
		return this->fstdistfunc_;
	}

	template<typename dist_t>
	inline void* IPSpace<dist_t>::get_dist_func_param() {
		return &this->dim_;
	}

	template<typename dist_t>
	inline IPSpace<dist_t>::IPSpace(const size_t dim) {
		this->fstdistfunc_ = innerProduct;
		this->dim_ = dim;
		this->data_size_ = dim * sizeof(dist_t);
	}

	template<typename dist_t>
	dist_t euclideanDist(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		dist_t* pVect1 = (dist_t*)pVect1v;
		dist_t* pVect2 = (dist_t*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);

		dist_t res = 0;
		for(size_t i = 0; i < qty; i++) {
			dist_t t = *pVect1 - *pVect2;
			pVect1++;
			pVect2++;
			res += t * t;
		}
		return (res);
	}

	template<typename dist_t>
	inline EuclideanSpace<dist_t>::EuclideanSpace(const size_t dim) {
		this->fstdistfunc_ = euclideanDist;
		this->dim_ = dim;
		this->data_size_ = dim * sizeof(dist_t);
	}

	template<typename dist_t>
	inline size_t EuclideanSpace<dist_t>::get_data_size() {
		return this->data_size_;
	}

	template<typename dist_t>
	inline hnswlib::DISTFUNC<dist_t> EuclideanSpace<dist_t>::get_dist_func() {
		return this->fstdistfunc_;
	}

	template<typename dist_t>
	inline void* EuclideanSpace<dist_t>::get_dist_func_param() {
		return &this->dim_;
	}
}
