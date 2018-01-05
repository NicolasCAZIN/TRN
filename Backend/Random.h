#pragma once

#include "backend_global.h"

namespace TRN
{
	namespace Backend
	{
		class BACKEND_EXPORT Random
		{
		protected:
			static const float DEFAULT_SPARSITY;
			static const float DEFAULT_A;
			static const float DEFAULT_B;
			static const float DEFAULT_MU;
			static const float DEFAULT_SIGMA;

		public:
			
			virtual void uniform(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const bool &blank_diagonal, const float &a = DEFAULT_A, const float &b = DEFAULT_B, const float &sparsity = DEFAULT_SPARSITY) = 0;
			virtual void gaussian(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *strides, const bool &blank_diagonal, const float &mu = DEFAULT_MU, const float &sigma = DEFAULT_SIGMA, const float &sparsity = DEFAULT_SPARSITY) = 0;
		};
	};
};

