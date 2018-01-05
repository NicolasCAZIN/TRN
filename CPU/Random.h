#pragma once

#include "cpu_global.h"
#include "Backend/Random.h"

namespace TRN
{
	namespace CPU
	{
		class CPU_EXPORT Random : public TRN::Backend::Random
		{
		public:
		
#ifdef CPU_LIB
		public :
			static void uniform_implementation( const unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const bool &blank_diagonal, const float &a, const float &b, const float &sparsity);
#endif

		public :
			virtual void uniform(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const bool &blank_diagonal, const float &a = DEFAULT_A, const float &b = DEFAULT_B, const float &sparsity = DEFAULT_SPARSITY) override;
			virtual void gaussian(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const bool &blank_diagonal, const float &mu = DEFAULT_MU, const float &sigma = DEFAULT_SIGMA, const float &sparsity = DEFAULT_SPARSITY) override;

		public :
			static std::shared_ptr<Random> create();
		};
	};
};
