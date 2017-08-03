#pragma once

#include "cpu_global.h"

namespace TRN
{
	namespace CPU
	{
		enum Implementation
		{
			SCALAR,
#if defined(_M_IX86) && !defined(_M_X64)
			MMX_SSE, //exp_sp compiled with MMX
#endif
#if !defined(_M_IX86) && defined(_M_X64)
			SSE2, //exp_sp compiled with SSE2
			SSE3, //hadd, exp_sp compiled with SSE2
			SSE41,// dp_ps, exp_ps compiled with SSE2
			AVX, // 256_dp_ps, exp256_ps compiled with SSE2
			AVX2, // exp256_ps compiled with avx2
			FMA3 //256_fmadd 
#endif

		};

		template<TRN::CPU::Implementation Implementation>
		struct  Traits
		{

		};

		template<>
		struct  Traits<SCALAR>
		{
			typedef float type;
			static const std::size_t step = 1;
		};
#if defined(_M_IX86) && !defined(_M_X64)
		template<>
		struct CPU_EXPORT Traits<MMX_SSE>
		{
			typedef __m128 type;
			static const std::size_t step = 4;
		};
#endif
#if !defined(_M_IX86) && defined(_M_X64)
		template<>
		struct  Traits<SSE2>
		{
			typedef __m128 type;
			static const std::size_t step = 4;
		};
		template<>
		struct  Traits<SSE3>
		{
			typedef __m128 type;
			static const std::size_t step = 4;
		};
		template<>
		struct  Traits<SSE41>
		{
			typedef __m128 type;
			static const std::size_t step = 4;
		};
		template<>
		struct  Traits<AVX>
		{
			typedef __m256 type;
			static const std::size_t step = 8;
		};
		template<>
		struct  Traits<AVX2>
		{
			typedef __m256 type;
			static const std::size_t step = 8;
		};
		template<>
		struct  Traits<FMA3>
		{
			typedef __m256 type;
			static const std::size_t step = 8;
		};
#endif
	};
};
