#pragma once
#include "chm/distances.hpp"
#include <hnswlib/hnswlib.h>
#include "KnnResults.hpp"

namespace hnswlibDist {
	static float InnerProduct(const void* pVect1, const void* pVect2, const void* qty_ptr) {
		size_t qty = *((size_t*)qty_ptr);
		float res = 0;
		for(unsigned i = 0; i < qty; i++)
			res += ((float*)pVect1)[i] * ((float*)pVect2)[i];
		return 1.f - res;
	}

	#if defined(USE_AVX)

	static float InnerProductSIMD4Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float PORTABLE_ALIGN32 TmpRes[8];
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);

		size_t qty16 = qty / 16;
		size_t qty4 = qty / 4;

		const float* pEnd1 = pVect1 + 16 * qty16;
		const float* pEnd2 = pVect1 + 4 * qty4;

		__m256 sum256 = _mm256_set1_ps(0);

		while(pVect1 < pEnd1) {
			__m256 v1 = _mm256_loadu_ps(pVect1);
			pVect1 += 8;
			__m256 v2 = _mm256_loadu_ps(pVect2);
			pVect2 += 8;
			sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

			v1 = _mm256_loadu_ps(pVect1);
			pVect1 += 8;
			v2 = _mm256_loadu_ps(pVect2);
			pVect2 += 8;
			sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
		}

		__m128 v1, v2;
		__m128 sum_prod = _mm_add_ps(_mm256_extractf128_ps(sum256, 0), _mm256_extractf128_ps(sum256, 1));

		while(pVect1 < pEnd2) {
			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
		}

		_mm_store_ps(TmpRes, sum_prod);
		float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];;
		return 1.f - sum;
	}

	#elif defined(USE_SSE)

	static float InnerProductSIMD4Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float PORTABLE_ALIGN32 TmpRes[8];
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);

		size_t qty16 = qty / 16;
		size_t qty4 = qty / 4;

		const float* pEnd1 = pVect1 + 16 * qty16;
		const float* pEnd2 = pVect1 + 4 * qty4;

		__m128 v1, v2;
		__m128 sum_prod = _mm_set1_ps(0);

		while(pVect1 < pEnd1) {
			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
		}

		while(pVect1 < pEnd2) {
			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
		}

		_mm_store_ps(TmpRes, sum_prod);
		float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

		return 1.f - sum;
	}

	#endif

	#if defined(USE_AVX512)

	static float InnerProductSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float PORTABLE_ALIGN64 TmpRes[16];
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);

		size_t qty16 = qty / 16;

		const float* pEnd1 = pVect1 + 16 * qty16;

		__m512 sum512 = _mm512_set1_ps(0);

		while(pVect1 < pEnd1) {
			__m512 v1 = _mm512_loadu_ps(pVect1);
			pVect1 += 16;
			__m512 v2 = _mm512_loadu_ps(pVect2);
			pVect2 += 16;
			sum512 = _mm512_add_ps(sum512, _mm512_mul_ps(v1, v2));
		}

		_mm512_store_ps(TmpRes, sum512);
		float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] + TmpRes[13] + TmpRes[14] + TmpRes[15];

		return 1.f - sum;
	}

	#elif defined(USE_AVX)

	static float InnerProductSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float PORTABLE_ALIGN32 TmpRes[8];
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);

		size_t qty16 = qty / 16;

		const float* pEnd1 = pVect1 + 16 * qty16;

		__m256 sum256 = _mm256_set1_ps(0);

		while(pVect1 < pEnd1) {
			__m256 v1 = _mm256_loadu_ps(pVect1);
			pVect1 += 8;
			__m256 v2 = _mm256_loadu_ps(pVect2);
			pVect2 += 8;
			sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));

			v1 = _mm256_loadu_ps(pVect1);
			pVect1 += 8;
			v2 = _mm256_loadu_ps(pVect2);
			pVect2 += 8;
			sum256 = _mm256_add_ps(sum256, _mm256_mul_ps(v1, v2));
		}

		_mm256_store_ps(TmpRes, sum256);
		float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];

		return 1.f - sum;
	}

	#elif defined(USE_SSE)

	static float InnerProductSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float PORTABLE_ALIGN32 TmpRes[8];
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);

		size_t qty16 = qty / 16;

		const float* pEnd1 = pVect1 + 16 * qty16;

		__m128 v1, v2;
		__m128 sum_prod = _mm_set1_ps(0);

		while(pVect1 < pEnd1) {
			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			sum_prod = _mm_add_ps(sum_prod, _mm_mul_ps(v1, v2));
		}

		_mm_store_ps(TmpRes, sum_prod);
		float sum = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];

		return 1.f - sum;
	}

	#endif

	#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)

	static float InnerProductSIMD16ExtResiduals(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		size_t qty = *((size_t*)qty_ptr);
		size_t qty16 = qty >> 4 << 4;
		float res = InnerProductSIMD16Ext(pVect1v, pVect2v, &qty16);
		float* pVect1 = (float*)pVect1v + qty16;
		float* pVect2 = (float*)pVect2v + qty16;

		size_t qty_left = qty - qty16;
		float res_tail = InnerProduct(pVect1, pVect2, &qty_left);
		return res + res_tail - 1.f;
	}

	static float InnerProductSIMD4ExtResiduals(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		size_t qty = *((size_t*)qty_ptr);
		size_t qty4 = qty >> 2 << 2;

		float res = InnerProductSIMD4Ext(pVect1v, pVect2v, &qty4);
		size_t qty_left = qty - qty4;

		float* pVect1 = (float*)pVect1v + qty4;
		float* pVect2 = (float*)pVect2v + qty4;
		float res_tail = InnerProduct(pVect1, pVect2, &qty_left);

		return res + res_tail - 1.f;
	}

	#endif

	static float L2Sqr(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);
		float res = 0;

		for(size_t i = 0; i < qty; i++) {
			float t = *pVect1 - *pVect2;
			pVect1++;
			pVect2++;
			res += t * t;
		}

		return res;
	}

	#if defined(USE_AVX512)

	static float L2SqrSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);
		float PORTABLE_ALIGN64 TmpRes[16];
		size_t qty16 = qty >> 4;

		const float* pEnd1 = pVect1 + (qty16 << 4);

		__m512 diff, v1, v2;
		__m512 sum = _mm512_set1_ps(0);

		while(pVect1 < pEnd1) {
			v1 = _mm512_loadu_ps(pVect1);
			pVect1 += 16;
			v2 = _mm512_loadu_ps(pVect2);
			pVect2 += 16;
			diff = _mm512_sub_ps(v1, v2);
			sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
		}

		_mm512_store_ps(TmpRes, sum);

		float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
			TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
			TmpRes[13] + TmpRes[14] + TmpRes[15];

		return res;
	}

	#elif defined(USE_AVX)

	static float L2SqrSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);
		float PORTABLE_ALIGN32 TmpRes[8];
		size_t qty16 = qty >> 4;

		const float* pEnd1 = pVect1 + (qty16 << 4);

		__m256 diff, v1, v2;
		__m256 sum = _mm256_set1_ps(0);

		while(pVect1 < pEnd1) {
			v1 = _mm256_loadu_ps(pVect1);
			pVect1 += 8;
			v2 = _mm256_loadu_ps(pVect2);
			pVect2 += 8;
			diff = _mm256_sub_ps(v1, v2);
			sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

			v1 = _mm256_loadu_ps(pVect1);
			pVect1 += 8;
			v2 = _mm256_loadu_ps(pVect2);
			pVect2 += 8;
			diff = _mm256_sub_ps(v1, v2);
			sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
		}

		_mm256_store_ps(TmpRes, sum);
		return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
	}

	#elif defined(USE_SSE)

	static float L2SqrSIMD16Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);
		float PORTABLE_ALIGN32 TmpRes[8];
		size_t qty16 = qty >> 4;

		const float* pEnd1 = pVect1 + (qty16 << 4);

		__m128 diff, v1, v2;
		__m128 sum = _mm_set1_ps(0);

		while(pVect1 < pEnd1) {
			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			diff = _mm_sub_ps(v1, v2);
			sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			diff = _mm_sub_ps(v1, v2);
			sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			diff = _mm_sub_ps(v1, v2);
			sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));

			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			diff = _mm_sub_ps(v1, v2);
			sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
		}

		_mm_store_ps(TmpRes, sum);
		return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
	}

	#endif

	#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)

	static float L2SqrSIMD16ExtResiduals(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		size_t qty = *((size_t*)qty_ptr);
		size_t qty16 = qty >> 4 << 4;
		float res = L2SqrSIMD16Ext(pVect1v, pVect2v, &qty16);
		float* pVect1 = (float*)pVect1v + qty16;
		float* pVect2 = (float*)pVect2v + qty16;

		size_t qty_left = qty - qty16;
		float res_tail = L2Sqr(pVect1, pVect2, &qty_left);
		return res + res_tail;
	}

	#endif

	#ifdef USE_SSE
	static float L2SqrSIMD4Ext(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		float PORTABLE_ALIGN32 TmpRes[8];
		float* pVect1 = (float*)pVect1v;
		float* pVect2 = (float*)pVect2v;
		size_t qty = *((size_t*)qty_ptr);

		size_t qty4 = qty >> 2;

		const float* pEnd1 = pVect1 + (qty4 << 2);

		__m128 diff, v1, v2;
		__m128 sum = _mm_set1_ps(0);

		while(pVect1 < pEnd1) {
			v1 = _mm_loadu_ps(pVect1);
			pVect1 += 4;
			v2 = _mm_loadu_ps(pVect2);
			pVect2 += 4;
			diff = _mm_sub_ps(v1, v2);
			sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
		}

		_mm_store_ps(TmpRes, sum);
		return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
	}

	static float L2SqrSIMD4ExtResiduals(const void* pVect1v, const void* pVect2v, const void* qty_ptr) {
		size_t qty = *((size_t*)qty_ptr);
		size_t qty4 = qty >> 2 << 2;

		float res = L2SqrSIMD4Ext(pVect1v, pVect2v, &qty4);
		size_t qty_left = qty - qty4;

		float* pVect1 = (float*)pVect1v + qty4;
		float* pVect2 = (float*)pVect2v + qty4;
		float res_tail = L2Sqr(pVect1, pVect2, &qty_left);

		return res + res_tail;
	}

	#endif
}

namespace chm {
	template<typename Dist>
	DistFunc<Dist> getDistFunc(const SpaceEnum s, const size_t dim);

	template<typename Dist>
	DistFunc<Dist> getDistFunc(const SpaceEnum s) {
		switch(s) {
			case SpaceEnum::ANGULAR:
			case SpaceEnum::INNER_PRODUCT:
				return innerProductDistance<Dist>;
			case SpaceEnum::EUCLIDEAN:
				return euclideanDistance<Dist>;
			default:
				throw std::runtime_error(UNKNOWN_SPACE);
		}
	}

	template<>
	DistFunc<float> getDistFunc(const SpaceEnum s, const size_t dim) {
		switch(s) {
			case SpaceEnum::ANGULAR:
			case SpaceEnum::INNER_PRODUCT:
				#if defined(USE_AVX) || defined(USE_SSE) || defined(USE_AVX512)

				if(dim % 16 == 0)
					return hnswlibDist::InnerProductSIMD16Ext;
				else if(dim % 4 == 0)
					return hnswlibDist::InnerProductSIMD4Ext;
				else if(dim > 16)
					return hnswlibDist::InnerProductSIMD16ExtResiduals;
				else if(dim > 4)
					return hnswlibDist::InnerProductSIMD4ExtResiduals;

				#endif

				return hnswlibDist::InnerProduct;
			case SpaceEnum::EUCLIDEAN:
				#if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)

				if(dim % 16 == 0)
					return hnswlibDist::L2SqrSIMD16Ext;
				else if(dim % 4 == 0)
					return hnswlibDist::L2SqrSIMD4Ext;
				else if(dim > 16)
					return hnswlibDist::L2SqrSIMD16ExtResiduals;
				else if(dim > 4)
					return hnswlibDist::L2SqrSIMD4ExtResiduals;
				#endif

				return hnswlibDist::L2Sqr;
			default:
				throw std::runtime_error(UNKNOWN_SPACE);
		}
	}
}
