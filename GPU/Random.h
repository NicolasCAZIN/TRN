#pragma once

#include "gpu_global.h"
#include "Context.h"
#include "Backend/Random.h"

namespace TRN
{
	namespace GPU
	{
		class GPU_EXPORT Random : public TRN::Backend::Random
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Random(const std::shared_ptr<Context> context);
			~Random();

		public:
			virtual void uniform(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *stride, const bool &blank_diagonal, const float &a = DEFAULT_A, const float &b = DEFAULT_B, const float &sparsity = DEFAULT_SPARSITY) override;
			virtual void gaussian(unsigned long &seed, float **ptr, const std::size_t &batch_size, const std::size_t *rows, const std::size_t *cols, const std::size_t *stride, const bool &blank_diagonal, const float &mu = DEFAULT_MU, const float &sigma = DEFAULT_SIGMA, const float &sparsity = DEFAULT_SPARSITY) override;

		public:
			static std::shared_ptr<Random> create(const std::shared_ptr<Context> context);
		};
	};
};
