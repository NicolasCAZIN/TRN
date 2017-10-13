#include "stdafx.h"

#if !defined(_M_IX86) && (defined(_M_AMD64) ||defined(_M_X64))
#include "Algorithm.h"
#define USE_SSE2
#include "sse_mathfun.h"



#define _0 0
#define _1 4
#define _2 8
#define _3 12
#define _4 16
#define _5 20
#define _6 24
#define _7 28
#define _8 32
#define _9 36
#define _10 40
#define _11 44
#define _12 48
#define _13 52
#define _14 56
#define _15 60
#define _16 64
#define _17 68
#define _18 72
#define _19 76
#define _20 80
#define _21 84
#define _22 88
#define _23 92
#define _24 96
#define _25 100
#define _26 104
#define _27 108
#define _28 112
#define _29 116
#define _30 120
#define _31 124
#define _32 128

#define get_element(k, __m) ((__m).m128_f32[(k)])
#define set_element(v, k, __m) ((__m).m128_f32[(k)] = (v))
#define set1_ps(s) (_mm_set1_ps((s)))
#define setzero_ps() (_mm_setzero_ps())
#define load_ps(S) (_mm_load_ps((S)))
#define stream_ps(S, __s) (_mm_stream_ps((S), (__s)))
#define stream_ss(S, s) (_mm_store_ss((S), _mm_set1_ps((s))))
#define add_ps(__a, __b) (_mm_add_ps((__a), (__b)))
#define sub_ps(__a, __b) (_mm_sub_ps((__a), (__b)))
#define mul_ps(__a, __b) (_mm_mul_ps((__a), (__b)))
static inline __m128 sqr_ps(const __m128 &__a)
{
	return mul_ps(__a, __a);
}
#define mul_add_ps(__a, __b, __c) (add_ps(mul_ps((__a), (__b)), (__c)))
static inline __m128 tanh_ps(const __m128 &__a)
{
	const v4sf one = _mm_set1_ps(1.0f);
	const v4sf two = _mm_set1_ps(2.0f);

	auto e = ::exp_ps(_mm_mul_ps(__a, two));
	return _mm_div_ps(_mm_sub_ps(e, one), _mm_add_ps(e, one));
}
#define exp_ps(__a) (::exp_ps(__a))
static inline float hsum_ps(const __m128 &__a)
{
	__m128 t1 = _mm_hadd_ps(__a, __a);
	return _mm_cvtss_f32(_mm_hadd_ps(t1, t1));
}



#define blendv_ps(__a, __b, __c) (_mm_blendv_ps((__a), (__b), (__c)))
#define cmp_lt_ps(__a, __b) (_mm_cmplt_ps((__a), (__b)))

#include "Algorithm_template_method.h"

/*template<>
static inline float  dot_product<TRN::CPU::Implementation::SSE41>(const float *x, const float *a, const std::size_t &cols)
{
	std::size_t col = 0;
	auto y0 = setzero_ps();
	if (cols - col > _16)
	{

		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();
		auto y4 = setzero_ps();
		auto y5 = setzero_ps();
		auto y6 = setzero_ps();
		auto y7 = setzero_ps();
		auto y8 = setzero_ps();
		auto y9 = setzero_ps();
		auto y10 = setzero_ps();
		auto y11 = setzero_ps();
		auto y12 = setzero_ps();
		auto y13 = setzero_ps();
		auto y14 = setzero_ps();
		auto y15 = setzero_ps();

		for (; col + _16 - 1 < cols; col += _16)
		{
			y0 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), 0xf1), y0);
			y1 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), 0xf2), y1);
			y2 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2]), 0xf4), y2);
			y3 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3]), 0xf8), y3);

			y4 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _4]), load_ps(&x[col + _4]), 0xf1), y4);
			y5 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _5]), load_ps(&x[col + _5]), 0xf2), y5);
			y6 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _6]), load_ps(&x[col + _6]), 0xf4), y6);
			y7 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _7]), load_ps(&x[col + _7]), 0xf8), y7);

			y8 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _8]), load_ps(&x[col + _8]), 0xf1), y8);
			y9 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _9]), load_ps(&x[col + _9]), 0xf2), y9);
			y10 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _10]), load_ps(&x[col + _10]), 0xf4), y10);
			y11 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _11]), load_ps(&x[col + _11]), 0xf8), y11);

			y12 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _12]), load_ps(&x[col + _12]), 0xf1), y12);
			y13 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _13]), load_ps(&x[col + _13]), 0xf2), y13);
			y14 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _14]), load_ps(&x[col + _14]), 0xf4), y14);
			y15 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _15]), load_ps(&x[col + _15]), 0xf8), y15);

		
		}

		y0 = _mm_or_ps(y0, y1);
		y2 = _mm_or_ps(y2, y3);
		y4 = _mm_or_ps(y4, y5);
		y6 = _mm_or_ps(y6, y7);

		y8 = _mm_or_ps(y8, y9);
		y10 = _mm_or_ps(y10, y11);
		y12 = _mm_or_ps(y12, y13);
		y14 = _mm_or_ps(y14, y15);

		y0 = _mm_add_ps(y0, y2);
		y4 = _mm_add_ps(y4, y6);
		y8 = _mm_add_ps(y8, y10);
		y12 = _mm_add_ps(y12, y14);

		y0 = _mm_add_ps(y0, y4);
		y8 = _mm_add_ps(y8, y12);
		y0 = _mm_add_ps(y0, y8);
	}

	if (cols - col > _8)
	{
		auto y0 = setzero_ps();
		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();
		auto y4 = setzero_ps();
		auto y5 = setzero_ps();
		auto y6 = setzero_ps();
		auto y7 = setzero_ps();

		for (; col + _8 - 1 < cols; col += _8)
		{
			y0 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), 0xf1), y0);
			y1 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), 0xf2), y1);
			y2 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2]), 0xf4), y2);
			y3 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3]), 0xf8), y3);
				
			y4 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _4]), load_ps(&x[col + _4]), 0xf1), y4);
			y5 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _5]), load_ps(&x[col + _5]), 0xf2), y5);
			y6 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _6]), load_ps(&x[col + _6]), 0xf4), y6);
			y7 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _7]), load_ps(&x[col + _7]), 0xf8), y7);
		}
		y0 = _mm_add_ps(y0, y1);
		y2 = _mm_add_ps(y2, y3);
		y4 = _mm_add_ps(y4, y5);
		y6 = _mm_add_ps(y6, y7);

		y0 = _mm_add_ps(y0, y2);
		y4 = _mm_add_ps(y4, y6);

		y0 = _mm_add_ps(y0, y4);
	}
	if (cols - col > _4)
	{
		auto y0 = setzero_ps();
		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();
		for (; col + _4 - 1 < cols; col += _4)
		{
			y0 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), 0xf1), y0);
			y1 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), 0xf2), y1);
			y2 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2]), 0xf4), y2);
			y3 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3]), 0xf8), y3);
		}
		y0 = _mm_add_ps(y0, y1);
		y2 = _mm_add_ps(y2, y3);
		y0 = _mm_add_ps(y0, y2);
	}
	if (cols - col > _2)
	{
		auto y0 = setzero_ps();
		auto y1 = setzero_ps();
		for (; col + _2 - 1 < cols; col += _2)
		{
			y0 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), 0xf1), y0);
			y1 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), 0xf2), y1);
		}
		y0 = _mm_add_ps(y0, y1);
	}
	if (cols - col > 0)
	{
		auto y0 = setzero_ps();
		for (; col < cols; col += _1)
		{
			y0 = _mm_add_ps(_mm_dp_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), 0xf1), y0);
		}
	}
	
	return hsum_ps(y0);
}
*/
template  TRN::CPU::Algorithm<TRN::CPU::Implementation::SSE41>;

#endif