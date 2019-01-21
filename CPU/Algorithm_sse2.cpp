#include "stdafx.h"

#if !defined(_M_IX86) && (defined(_M_AMD64) ||defined(_M_X64))
#include "Algorithm.h"
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
#define div_ps(__a, __b) (_mm_div_ps((__a), (__b)))
#define rsqrt_ps(__a) (_mm_rsqrt_ps((__a)))
#define and_ps(__a, __b) (_mm_and_ps((__a), (__b)))
#define or_ps(__a, __b) (_mm_or_ps((__a), (__b)))
#define abs_ps(__a) (_mm_andnot_ps(_mm_castsi128_ps(_mm_set1_epi32(0x80000000)), (__a)))

static inline __m128 sqr_ps(const __m128 &__a)
{
	return mul_ps(__a, __a);
}
#define mul_add_ps(__a, __b, __c) (add_ps(mul_ps((__a), (__b)), (__c)))
static inline float hsum_ps(const __m128 &__a)
{
	__m128 shuf = _mm_shuffle_ps(__a, __a, _MM_SHUFFLE(2, 3, 0, 1));  // [ C D | A B ]
	__m128 sums = _mm_add_ps(__a, shuf);      // sums = [ D+C C+D | B+A A+B ]
	shuf = _mm_movehl_ps(shuf, sums);      //  [   C   D | D+C C+D ]  // let the compiler avoid a mov by reusing shuf
	sums = _mm_add_ss(sums, shuf);
	return    _mm_cvtss_f32(sums);
}

#define blendv_ps(__a, __b, __c) (_mm_blendv_ps((__a), (__b), (__c)))
#define cmp_lt_ps(__a, __b) (_mm_cmplt_ps((__a), (__b)))
#define cmp_eq_ps(__a, __b) (_mm_cmpeq_ps((__a), (__b)))
#define exp_ps(__a) (::exp_ps((__a)))

#include "Algorithm_template_method.h"

template  TRN::CPU::Algorithm<TRN::CPU::Implementation::SSE2>;

#endif 