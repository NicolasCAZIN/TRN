#pragma once

#include "model_global.h"

#include "Core/Measurement.h"

namespace TRN
{
	namespace Model
	{
		namespace Measurement
		{
			namespace MeanSquareError
			{
				std::shared_ptr<TRN::Core::Measurement::Implementation> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor);
			};

			namespace FrechetDistance
			{
				std::shared_ptr<TRN::Core::Measurement::Implementation> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols)> &functor);
			};

			namespace Custom
			{
				std::shared_ptr<TRN::Core::Measurement::Implementation> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::function<void(const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &pages,  const std::size_t &rows, const  std::size_t &cols)> &functor);
			};

			namespace Position
			{
				std::shared_ptr<TRN::Core::Measurement::Abstraction> MODEL_EXPORT create(const std::shared_ptr<TRN::Core::Measurement::Implementation> &implementation, const std::size_t &batch_size);
			};

			namespace Sequence
			{
				std::shared_ptr<TRN::Core::Measurement::Abstraction> MODEL_EXPORT create(const std::shared_ptr<TRN::Core::Measurement::Implementation> &implementation, const std::size_t &batch_size);
			};
		};
	};
};