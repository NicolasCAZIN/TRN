#pragma once
/*!
@file sse_mathfun.h
SIMD (SSE1+MMX or SSE2) implementation of sin, cos, exp and log
Inspired by Intel Approximate Math library, and based on the
corresponding algorithms of the cephes math library
The default is to use the SSE1 version. If you define USE_SSE2 the
the SSE2 intrinsics will be used in place of the MMX intrinsics. Do
not expect any significant performance improvement with SSE2.
*/

/* Copyright (C) 2010,2011  RJVB - extensions */
/* Copyright (C) 2007  Julien Pommier
This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.
Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:
1. The origin of this software must not be misrepresented; you must not
claim that you wrote the original software. If you use this software
in a product, an acknowledgment in the product documentation would be
appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
(this is the zlib license)
*/

#ifndef _SSE_MATHFUN_H

#ifdef USE_SSE_AUTO
#	ifdef __SSE2__
#		if defined(__GNUC__)
#			warning "USE_SSE2"
#		endif
#		define USE_SSE2
#	endif
#	if defined(__SSE3__) || defined(__SSSE3__)
#		if defined(__GNUC__)
#			warning "USE_SSE3"
#		endif
#		define USE_SSE2
#		define USE_SSE3
#	endif
#	if defined(__SSE4__) || defined(__SSE4_1__) || defined(__SSE4_2__) || ((_M_IX86_FP > 1) && !defined(_M_AMD64))
#		if defined(__GNUC__)
#			warning "USE_SSE4"
#		endif
#		define USE_SSE2
#		define USE_SSE3
#		define USE_SSE4
#	endif
#endif

#include <math.h>
#include <xmmintrin.h>
#include <emmintrin.h>

/* yes I know, the top of this file is quite ugly */

/*!
macros to obtain the required 16bit alignment
*/
#ifdef _MSC_VER /* visual c++ */
# define ALIGN16_BEG __declspec(align(16))
# define ALIGN16_END
# define inline	__forceinline
#else /* gcc or icc */
# define ALIGN16_BEG
# define ALIGN16_END __attribute__((aligned(16)))
#endif

/* __m128 is ugly to write */
/*!
an SSE vector of 4 floats
*/
typedef __m128 v4sf;  // vector of 4 float (sse1)

#if defined(USE_SSE3) || defined(USE_SSE4)
#	define USE_SSE2
#endif

					  /*!
					  an SSE/MMX vector of 4 32bit integers
					  */
#ifdef __APPLE_CC__
typedef int	v4si __attribute__((__vector_size__(16), __may_alias__));
#else
typedef __m128i v4si; // vector of 4 int (sse2)
#endif
					  // RJVB 20111028: some support for double precision semantics
					  /*!
					  an SSE2+ vector of 2 doubles
					  */
typedef __m128d v2df; // vector of 2 double (sse2)
					  /*!
					  an MMX vector of 2 32bit ints
					  */
typedef __m64 v2si;   // vector of 2 int (mmx)

#if defined(USE_SSE3) || defined(USE_SSE4)
#	define USE_SSE3
#	include <pmmintrin.h>
#	if defined(__SSSE3__) || (_M_IX86_FP > 1)
#		include <tmmintrin.h>
#	endif
#endif

#if defined(USE_SSE4)
#	define USE_SSE4
#	include <smmintrin.h>
#endif

#ifdef __GNUC__0
#	define _MM_SET_PD(b,a)		(v2df){(a),(b)}
#	define _MM_SET1_PD(a)		(v2df){(a),(a)}
					  // 	static inline v2df _MM_SET1_PD(double a)
					  // 	{
					  // 		return (v2df){a,a};
					  // 	}
#	define _MM_SETR_PD(a,b)		(v2df){(a),(b)}
#	define _MM_SETZERO_PD()		(v2df){0.0,0.0}
#	define _MM_SET_PS(d,c,b,a)	(v4sf){(a),(b),(c),(d)}
#	define _MM_SET1_PS(a)		(v4sf){(a),(a),(a),(a)}
					  // 	static inline v4sf _MM_SET1_PS(float a)
					  // 	{
					  // 		return (v4sf){a,a,a,a};
					  // 	}
#	define _MM_SETR_PS(a,b,c,d)	(v4sf){(a),(b),(c),(d)}
#	define _MM_SETZERO_PS()		(v4sf){0.0f,0.0f,0.0f,0.0f}
#	define _MM_SETZERO_SI128()	(__m128i)(__v4si){0,0,0,0}
#	define _MM_SETZERO_SI64()	ALIGN16_BEG (__m64 ALIGN16_END)0LL
#else
#	define _MM_SET_PD(b,a)		_mm_setr_pd((a),(b))
#	define _MM_SET1_PD(a)		_mm_set1_pd((a))
#	define _MM_SETR_PD(a,b)		_mm_setr_pd((a),(b))
#	define _MM_SETZERO_PD()		_mm_setzero_pd()
#	define _MM_SET_PS(d,c,b,a)	_mm_setr_ps((a),(b),(c),(d))
#	define _MM_SET1_PS(a)		_mm_set1_ps((a))
#	define _MM_SETR_PS(a,b,c,d)	_mm_setr_ps((a),(b),(c),(d))
#	define _MM_SETZERO_PS()		_mm_setzero_ps()
#	define _MM_SETZERO_SI128()	_mm_setzero_si128()
#	define _MM_SETZERO_SI64()	_mm_setzero_si64()
#endif
#define VELEM(type,a,n)			(((type*)&a)[n])

					  /* declare some SSE constants -- why can't I figure a better way to do that? */
#define _PS_CONST(Name, Val)                                            \
  static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = { (const float)(Val), (const float)(Val), (const float)(Val), (const float)(Val) }
#define _PI32_CONST(Name, Val)                                            \
  static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = { Val, Val, Val, Val }
#define _PS_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = { Val, Val, Val, Val }

#define _PD_CONST(Name, Val)                                            \
	static const ALIGN16_BEG double _pd_##Name[2] ALIGN16_END = { (const double)(Val), (const double)(Val) }
#define _PD_CONST_TYPE(Name, Type, Val)                                 \
	static const ALIGN16_BEG Type _pd_##Name[2] ALIGN16_END = { Val, Val }

#ifdef SSE_MATHFUN_WITH_CODE

_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, 0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524);
_PS_CONST(cephes_log_p0, 7.0376836292E-2);
_PS_CONST(cephes_log_p1, -1.1514610310E-1);
_PS_CONST(cephes_log_p2, 1.1676998740E-1);
_PS_CONST(cephes_log_p3, -1.2420140846E-1);
_PS_CONST(cephes_log_p4, +1.4249322787E-1);
_PS_CONST(cephes_log_p5, -1.6668057665E-1);
_PS_CONST(cephes_log_p6, +2.0000714765E-1);
_PS_CONST(cephes_log_p7, -2.4999993993E-1);
_PS_CONST(cephes_log_p8, +3.3333331174E-1);
_PS_CONST(cephes_log_q1, -2.12194440e-4);
_PS_CONST(cephes_log_q2, 0.693359375);

#ifdef USE_SSE2
_PD_CONST(1, 1.0);
_PD_CONST(_1, -1.0);
_PD_CONST(0p5, 0.5);
/* the smallest non denormalised float number */
//	_PD_CONST_TYPE(min_norm_pos, int, 0x00800000);
//	_PD_CONST_TYPE(mant_mask, int, 0x7f800000);
//	_PD_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PD_CONST_TYPE(sign_mask, long long, 0x8000000000000000LL);
_PD_CONST_TYPE(inv_sign_mask, long long, ~0x8000000000000000LL);

#endif

#if defined (__MINGW32__)

/* the ugly part below: many versions of gcc used to be completely buggy with respect to some intrinsics
The movehl_ps is fixed in mingw 3.4.5, but I found out that all the _mm_cmp* intrinsics were completely
broken on my mingw gcc 3.4.5 ...
Note that the bug on _mm_cmp* does occur only at -O0 optimization level
*/

inline __m128 my_movehl_ps(__m128 a, const __m128 b) {
	asm(
		"movhlps %2,%0\n\t"
		: "=x" (a)
		: "0" (a), "x"(b)
	);
	return a;
}
#warning "redefined _mm_movehl_ps (see gcc bug 21179)"
#define _mm_movehl_ps my_movehl_ps

inline __m128 my_cmplt_ps(__m128 a, const __m128 b) {
	asm(
		"cmpltps %2,%0\n\t"
		: "=x" (a)
		: "0" (a), "x"(b)
	);
	return a;
}
inline __m128 my_cmpgt_ps(__m128 a, const __m128 b) {
	asm(
		"cmpnleps %2,%0\n\t"
		: "=x" (a)
		: "0" (a), "x"(b)
	);
	return a;
}
inline __m128 my_cmpeq_ps(__m128 a, const __m128 b) {
	asm(
		"cmpeqps %2,%0\n\t"
		: "=x" (a)
		: "0" (a), "x"(b)
	);
	return a;
}
#warning "redefined _mm_cmpxx_ps functions..."
#define _mm_cmplt_ps my_cmplt_ps
#define _mm_cmpgt_ps my_cmpgt_ps
#define _mm_cmpeq_ps my_cmpeq_ps
#endif

#ifndef USE_SSE2
typedef union xmm_mm_union {
	__m128 xmm;
	__m64 mm[2];
} xmm_mm_union;

#define COPY_XMM_TO_MM(xmm_, mm0_, mm1_) {          \
    xmm_mm_union u; u.xmm = xmm_;                   \
    mm0_ = u.mm[0];                                 \
    mm1_ = u.mm[1];                                 \
}

#define COPY_MM_TO_XMM(mm0_, mm1_, xmm_) {                         \
    xmm_mm_union u; u.mm[0]=mm0_; u.mm[1]=mm1_; xmm_ = u.xmm;      \
  }

#endif // USE_SSE2

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341);
_PS_CONST(cephes_exp_C1, 0.693359375);
_PS_CONST(cephes_exp_C2, -2.12194440e-4);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1);

/*!
computes e**x of the 4 floats in x
*/
static inline v4sf exp_ps(v4sf x)
{
	v4sf tmp = _MM_SETZERO_PS(), fx, mask, y, z;
	v4sf pow2n;
#ifdef USE_SSE2
	v4si emm0;
#else
	v2si mm0, mm1;
#endif
	v4sf one = *(v4sf*)_ps_1;

	x = _mm_min_ps(x, *(v4sf*)_ps_exp_hi);
	x = _mm_max_ps(x, *(v4sf*)_ps_exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	fx = _mm_mul_ps(x, *(v4sf*)_ps_cephes_LOG2EF);
	fx = _mm_add_ps(fx, *(v4sf*)_ps_0p5);

	/* how to perform a floorf with SSE: just below */
#ifndef USE_SSE2
	/* step 1 : cast to int */
	tmp = _mm_movehl_ps(tmp, fx);
	mm0 = _mm_cvttps_pi32(fx);
	mm1 = _mm_cvttps_pi32(tmp);
	/* step 2 : cast back to float */
	tmp = _mm_cvtpi32x2_ps(mm0, mm1);
#else
	emm0 = _mm_cvttps_epi32(fx);
	tmp = _mm_cvtepi32_ps(emm0);
#endif
	/* if greater, substract 1 */
	mask = _mm_cmpgt_ps(tmp, fx);
	mask = _mm_and_ps(mask, one);
	fx = _mm_sub_ps(tmp, mask);

	tmp = _mm_mul_ps(fx, *(v4sf*)_ps_cephes_exp_C1);
	z = _mm_mul_ps(fx, *(v4sf*)_ps_cephes_exp_C2);
	x = _mm_sub_ps(x, tmp);
	x = _mm_sub_ps(x, z);

	z = _mm_mul_ps(x, x);

	y = *(v4sf*)_ps_cephes_exp_p0;
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p1);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p2);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p3);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p4);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(v4sf*)_ps_cephes_exp_p5);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, x);
	y = _mm_add_ps(y, one);

	/* build 2^n */
#ifndef USE_SSE2
	z = _mm_movehl_ps(z, fx);
	mm0 = _mm_cvttps_pi32(fx);
	mm1 = _mm_cvttps_pi32(z);
	mm0 = _mm_add_pi32(mm0, *(v2si*)_pi32_0x7f);
	mm1 = _mm_add_pi32(mm1, *(v2si*)_pi32_0x7f);
	mm0 = _mm_slli_pi32(mm0, 23);
	mm1 = _mm_slli_pi32(mm1, 23);

	COPY_MM_TO_XMM(mm0, mm1, pow2n);
	_mm_empty();
#else
	emm0 = _mm_cvttps_epi32(fx);
	emm0 = _mm_add_epi32(emm0, *(v4si*)_pi32_0x7f);
	emm0 = _mm_slli_epi32(emm0, 23);
	pow2n = _mm_castsi128_ps(emm0);
#endif
	y = _mm_mul_ps(y, pow2n);
	return y;
}

_PS_CONST(minus_cephes_DP1, -0.78515625);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PS_CONST(sincof_p0, -1.9515295891E-4);
_PS_CONST(sincof_p1, 8.3321608736E-3);
_PS_CONST(sincof_p2, -1.6666654611E-1);
_PS_CONST(coscof_p0, 2.443315711809948E-005);
_PS_CONST(coscof_p1, -1.388731625493765E-003);
_PS_CONST(coscof_p2, 4.166664568298827E-002);
_PS_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI

#ifdef USE_SSE2
_PD_CONST(minus_cephes_DP1, -0.78515625);
_PD_CONST(minus_cephes_DP2, -2.4187564849853515625e-4);
_PD_CONST(minus_cephes_DP3, -3.77489497744594108e-8);
_PD_CONST(sincof_p0, -1.9515295891E-4);
_PD_CONST(sincof_p1, 8.3321608736E-3);
_PD_CONST(sincof_p2, -1.6666654611E-1);
_PD_CONST(coscof_p0, 2.443315711809948E-005);
_PD_CONST(coscof_p1, -1.388731625493765E-003);
_PD_CONST(coscof_p2, 4.166664568298827E-002);
_PD_CONST(cephes_FOPI, 1.27323954473516); // 4 / M_PI
#endif






#endif // SSE_MATHFUN_WITH_CODE

//// Some SSE "extensions", and equivalents not using SSE explicitly:





#define _SSE_MATHFUN_H
#endif
