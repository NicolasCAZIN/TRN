#include "stdafx.h"

#if !defined(_M_IX86) && (defined(_M_AMD64) ||defined(_M_X64))
#include "Algorithm.h"

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
#define get_element(k, __m) ((__m))
#define set_element(v, k, __m) ((__m) = (v))
#define set1_ps(s) ((s))
#define setzero_ps() (0.0f)
#define load_ps(S) ((*S))
#define stream_ps(S, __s) ((*S) = (__s))
#define stream_ss(S, s) ((*S) = (s))
#define add_ps(__a, __b) ((__a) + (__b))
#define sub_ps(__a, __b) ((__a) - (__b))
#define mul_ps(__a, __b) ((__a) * (__b))
#define div_ps(__a, __b) ((__a) / (__b))
#define sqr_ps(__a) (mul_ps((__a), (__a)))
#define mul_add_ps(__a, __b, __c) (add_ps(mul_ps((__a), (__b)), (__c)))
//#define tanh_ps(__a) (tanhf(__a))

#define dp_ps(__a, __b) 	(add_ps(mul_ps((__a), (__b))))
#define hsum_ps(__a) ((__a))

#define blendv_ps(__a, __b, __c) ((__c) ? (__a) : (__b))

#define cmp_lt_ps(__a, __b) ((__a) < (__b)) 

#define cmp_eq_ps(__a, __b) ((__a) == (__b)) 
#define rsqrt_ps(__a) ((1.0f/sqrtf((__a))))
#define and_ps(__a, __b) ((__a) && (__b))
#define or_ps(__a, __b) ((__a) || (__b))

#define abs_ps(__a) (std::fabs((__a)))
#define exp_ps(__a) (std::expf((__a)))
#include "Algorithm_template_method.h"

template  TRN::CPU::Algorithm<TRN::CPU::Implementation::SCALAR>;

#endif