#include "stdafx.h"

/*#if !defined(_M_IX86) && defined(_M_X64)
#include "Algorithm.h"
#define __AVX2__
#include "avx_mathfun.h"


static const v8sf one = _mm256_set1_ps(1.0f);
static const v8sf two = _mm256_set1_ps(2.0f);

#define _0 0
#define _1 8
#define _2 16
#define _3 24
#define _4 32
#define _5 40
#define _6 48
#define _7 56
#define _8 64
#define _9 72
#define _10 80
#define _11 88
#define _12 96
#define _13 104
#define _14 112
#define _15 120
#define _16 128
#define _17 136
#define _18 144
#define _19 152
#define _20 160
#define _21 168
#define _22 176
#define _23 184
#define _24 192
#define _25 200
#define _26 208
#define _27 216
#define _28 224
#define _29 232
#define _30 240
#define _31 248
#define _32 256

#define get_element(k, __m) ((__m).m256_f32[(k)])
#define set_element(v, k, __m) ((__m).m256_f32[(k)] = (v))
#define set1_ps(s) (_mm256_set1_ps((s)))
#define setzero_ps() (_mm256_setzero_ps())
#define load_ps(S) (_mm256_load_ps((S)))
#define stream_ps(S, __s) (_mm256_stream_ps((S), (__s)))
#define add_ps(__a, __b) (_mm256_add_ps((__a), (__b)))
#define sub_ps(__a, __b) (_mm256_sub_ps((__a), (__b)))
#define mul_ps(__a, __b) (_mm256_mul_ps((__a), (__b)))
#define sqr_ps(__a) (mul_ps((__a), (__a)))
#define mul_add_ps(__a, __b, __c) (_mm256_fmadd_ps((__a), (__b), (__c)))



static inline __m256 tanh_ps(const __m256 &__a)
{
	auto e = ::exp256_ps(_mm256_mul_ps(__a, two));
	return _mm256_div_ps(_mm256_sub_ps(e, one), _mm256_add_ps(e, one));
}
#define exp_ps(__a) (::exp256_ps(__a))
#define dp_acc_ps(__a, __b, __c) 	((__c) =mul_add_ps((__a), (__b), (__c)))
static inline float hsum_ps(const __m256 &__a)
{
	__m256 t1 = _mm256_hadd_ps((__a), (__a));
	__m256 t2 = _mm256_hadd_ps(t1, t1);
	float s = _mm_cvtss_f32(_mm_add_ss(_mm256_extractf128_ps(t2, 0), _mm256_extractf128_ps(t2, 1)));
	return s ;
}
#define blendv_ps(__a, __b, __c) (_mm256_blendv_ps((__a), (__b), (__c)))
#define cmp_ps(__a, __b, __c) (_mm256_cmp_ps((__a), (__b), (__c)))

#include "Algorithm_template_method.h"

template  TRN::CPU::Algorithm<TRN::CPU::Implementation::FMA3>;

#endif */