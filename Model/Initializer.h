#pragma once

#include "model_global.h"

#include "Core/Initializer.h"

namespace TRN
{
	namespace Model
	{
		namespace Initializer
		{
			namespace Uniform
			{
				std::shared_ptr<TRN::Core::Initializer> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver,  const float &a, const float &b, const float &sparsity);
			};
			namespace Gaussian
			{
				std::shared_ptr<TRN::Core::Initializer> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, const float &mu, const float &sigma, const float &sparsity);
			};
			namespace Custom
			{
				std::shared_ptr<TRN::Core::Initializer> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, 
					const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
					std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
			};
		};
	};
}; 