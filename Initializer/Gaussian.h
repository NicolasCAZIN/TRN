#pragma once

#include "initializer_global.h"
#include "Core/Initializer.h"

namespace TRN
{
	namespace Initializer
	{
		class INITIALIZER_EXPORT Gaussian : public TRN::Core::Initializer
		{
		private:
			class Handle;
			mutable std::unique_ptr<Handle> handle;

		public:
			Gaussian(const std::shared_ptr<TRN::Backend::Driver> &driver, const float &mu, const float &sigma);
			virtual ~Gaussian();

		public:
			virtual void initialize(unsigned long &seed, std::shared_ptr<TRN::Core::Batch> &batch) override;

		public:
			static std::shared_ptr<Gaussian> create(const std::shared_ptr<TRN::Backend::Driver> &driver,  const float &mu, const float &sigma);
		};
	};
};

