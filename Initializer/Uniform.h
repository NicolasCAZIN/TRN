#pragma once

#include "initializer_global.h"
#include "Core/Initializer.h"

namespace TRN
{
	namespace Initializer
	{
		class INITIALIZER_EXPORT Uniform : public TRN::Core::Initializer
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Uniform(const std::shared_ptr<TRN::Backend::Driver> &driver,const float &a, const float &b, const float &sparsity);
			virtual ~Uniform();

		public:
			virtual void initialize(unsigned long &seed, std::shared_ptr<TRN::Core::Batch> &batch, const bool &blank_diagonal) override;

		public:
			static std::shared_ptr<Uniform> create(const std::shared_ptr<TRN::Backend::Driver> &driver,  const float &a, const float &b, const float &sparsity);
		};
	};
};

