#pragma once

#include "cpu_global.h"

namespace TRN
{
	namespace CPU
	{
		enum CPU_EXPORT Implementation
		{
			SCALAR,
#if defined(_M_IX86) && !defined(_M_X64)
			MMX_SSE, //exp_sp compiled with MMX
#endif
#if !defined(_M_IX86) && defined(_M_X64)
			SSE2,
			SSE3, 
			SSE41,
			AVX, 
			AVX2_FMA3
#endif

		};


		void CPU_EXPORT query(std::string &brand, TRN::CPU::Implementation &implementation);

		template<TRN::CPU::Implementation Implementation>
		struct  Traits
		{

		};

		template<>
		struct CPU_EXPORT  Traits<SCALAR>
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
		struct CPU_EXPORT Traits<SSE2>
		{
			typedef __m128 type;
			static const std::size_t step = 4;
		};
		template<>
		struct CPU_EXPORT Traits<SSE3>
		{
			typedef __m128 type;
			static const std::size_t step = 4;
		};
		template<>
		struct CPU_EXPORT Traits<SSE41>
		{
			typedef __m128 type;
			static const std::size_t step = 4;
		};
		template<>
		struct  CPU_EXPORT Traits<AVX>
		{
			typedef __m256 type;
			static const std::size_t step = 8;
		};
		template<>
		struct CPU_EXPORT Traits<AVX2_FMA3>
		{
			typedef __m256 type;
			static const std::size_t step = 8;
		};
#endif
	};
};


std::ostream CPU_EXPORT &operator << (std::ostream &ostream, const TRN::CPU::Implementation &implementation);
