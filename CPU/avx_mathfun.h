#pragma once
/*
AVX implementation of sin, cos, sincos, exp and log

Based on "sse_mathfun.h", by Julien Pommier
http://gruntthepeon.free.fr/ssemath/

Copyright (C) 2012 Giovanni Garberoglio
Interdisciplinary Laboratory for Computational Science (LISC)
Fondazione Bruno Kessler and University of Trento
via Sommarive, 18
I-38123 Trento (Italy)

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

#include <immintrin.h>

/* yes I know, the top of this file is quite ugly */
#ifdef _MSC_VER /* visual c++ */
# define ALIGN32_BEG __declspec(align(32))
# define ALIGN32_END
# define inline	__forceinline
#else /* gcc or icc */
# define ALIGN32_BEG
# define ALIGN32_END __attribute__((aligned(32)))
#endif

/* __m128 is ugly to write */
typedef __m256  v8sf; // vector of 8 float (avx)
typedef __m256i v8si; // vector of 8 int   (avx)
typedef __m128i v4si; // vector of 8 int   (avx)

#define _PI32AVX_CONST(Name, Val)                                            \
  static const ALIGN32_BEG int _pi32avx_##Name[4] ALIGN32_END = { Val, Val, Val, Val }

_PI32AVX_CONST(1, 1);
_PI32AVX_CONST(inv1, ~1);
_PI32AVX_CONST(2, 2);
_PI32AVX_CONST(4, 4);


/* declare some AVX constants -- why can't I figure a better way to do that? */
#define _PS256_CONST(Name, Val)                                            \
  static const ALIGN32_BEG float _ps256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PI32_CONST256(Name, Val)                                            \
  static const ALIGN32_BEG int _pi32_256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }
#define _PS256_CONST_TYPE(Name, Type, Val)                                 \
  static const ALIGN32_BEG Type _ps256_##Name[8] ALIGN32_END = { Val, Val, Val, Val, Val, Val, Val, Val }

_PS256_CONST(1, 1.0f);
_PS256_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS256_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS256_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS256_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS256_CONST_TYPE(sign_mask, int, 0x80000000);
_PS256_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST256(0, 0);
_PI32_CONST256(1, 1);
_PI32_CONST256(inv1, ~1);
_PI32_CONST256(2, 2);
_PI32_CONST256(4, 4);
_PI32_CONST256(0x7f, 0x7f);



#ifndef __AVX2__

typedef union imm_xmm_union {
	v8si imm;
	v4si xmm[2];
} imm_xmm_union;

#define COPY_IMM_TO_XMM(imm_, xmm0_, xmm1_) {    \
    imm_xmm_union u ALIGN32_BEG;  \
    u.imm = imm_;				   \
    xmm0_ = u.xmm[0];                            \
    xmm1_ = u.xmm[1];                            \
}

#define COPY_XMM_TO_IMM(xmm0_, xmm1_, imm_) {                       \
    imm_xmm_union u ALIGN32_BEG; \
    u.xmm[0]=xmm0_; u.xmm[1]=xmm1_; imm_ = u.imm; \
  }


#define AVX2_BITOP_USING_SSE2(fn) \
static inline v8si _mm256_##fn(v8si x, int a) \
{ \
  /* use SSE2 instruction to perform the bitop AVX2 */ \
  v4si x1, x2; \
  v8si ret; \
  COPY_IMM_TO_XMM(x, x1, x2); \
  x1 = _mm_##fn(x1,a); \
  x2 = _mm_##fn(x2,a); \
  COPY_XMM_TO_IMM(x1, x2, ret); \
  return(ret); \
}


/*AVX2_BITOP_USING_SSE2(slli_epi32)
AVX2_BITOP_USING_SSE2(srli_epi32)*/

#define AVX2_INTOP_USING_SSE2(fn) \
static inline v8si _mm256_##fn(v8si x, v8si y) \
{ \
  /* use SSE2 instructions to perform the AVX2 integer operation */ \
  v4si x1, x2; \
  v4si y1, y2; \
  v8si ret; \
  COPY_IMM_TO_XMM(x, x1, x2); \
  COPY_IMM_TO_XMM(y, y1, y2); \
  x1 = _mm_##fn(x1,y1); \
  x2 = _mm_##fn(x2,y2); \
  COPY_XMM_TO_IMM(x1, x2, ret); \
  return(ret); \
}

//#warning "Using SSE2 to perform AVX2 integer ops"
/*AVX2_INTOP_USING_SSE2(and_si128)
AVX2_INTOP_USING_SSE2(andnot_si128)
AVX2_INTOP_USING_SSE2(cmpeq_epi32)
AVX2_INTOP_USING_SSE2(sub_epi32)
AVX2_INTOP_USING_SSE2(add_epi32)*/

#endif /* __AVX2__ */

_PS256_CONST(exp_hi, 88.3762626647949f);
_PS256_CONST(exp_lo, -88.3762626647949f);

_PS256_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS256_CONST(cephes_exp_C1, 0.693359375f);
_PS256_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS256_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS256_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS256_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS256_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS256_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS256_CONST(cephes_exp_p5, 5.0000001201E-1f);

static inline v8sf exp256_ps(v8sf x) {
	v8sf tmp = _mm256_setzero_ps(), fx;
	v8si imm0;
	v8sf one = *(v8sf*)_ps256_1;

	x = _mm256_min_ps(x, *(v8sf*)_ps256_exp_hi);
	x = _mm256_max_ps(x, *(v8sf*)_ps256_exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	fx = _mm256_mul_ps(x, *(v8sf*)_ps256_cephes_LOG2EF);
	fx = _mm256_add_ps(fx, *(v8sf*)_ps256_0p5);

	/* how to perform a floorf with SSE: just below */
	//imm0 = _mm256_cvttps_epi32(fx);
	//tmp  = _mm256_cvtepi32_ps(imm0);

	tmp = _mm256_floor_ps(fx);

	/* if greater, substract 1 */
	//v8sf mask = _mm256_cmpgt_ps(tmp, fx);    
	v8sf mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);
	mask = _mm256_and_ps(mask, one);
	fx = _mm256_sub_ps(tmp, mask);

	tmp = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C1);
	v8sf z = _mm256_mul_ps(fx, *(v8sf*)_ps256_cephes_exp_C2);
	x = _mm256_sub_ps(x, tmp);
	x = _mm256_sub_ps(x, z);

	z = _mm256_mul_ps(x, x);

	v8sf y = *(v8sf*)_ps256_cephes_exp_p0;
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p1);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p2);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p3);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p4);
	y = _mm256_mul_ps(y, x);
	y = _mm256_add_ps(y, *(v8sf*)_ps256_cephes_exp_p5);
	y = _mm256_mul_ps(y, z);
	y = _mm256_add_ps(y, x);
	y = _mm256_add_ps(y, one);

	/* build 2^n */
	imm0 = _mm256_cvttps_epi32(fx);
	// another two AVX2 instructions
	imm0 = _mm256_add_epi32(imm0, *(v8si*)_pi32_256_0x7f);
	imm0 = _mm256_slli_epi32(imm0, 23);
	v8sf pow2n = _mm256_castsi256_ps(imm0);
	y = _mm256_mul_ps(y, pow2n);
	return y;
}





