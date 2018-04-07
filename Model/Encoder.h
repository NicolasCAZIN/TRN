#pragma once

#include "model_global.h"

#include "Core/Encoder.h"

namespace TRN
{
	namespace Model
	{
		namespace Encoder
		{
			namespace Model
			{
				std::shared_ptr<TRN::Core::Encoder> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver,
						const std::size_t &batch_size, const std::size_t &stimulus_size,
						const std::shared_ptr<TRN::Core::Matrix> &cx,
						const std::shared_ptr<TRN::Core::Matrix> &cy,
						const std::shared_ptr<TRN::Core::Matrix> &K);
			};


			namespace Custom
			{
				std::shared_ptr<TRN::Core::Encoder> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver,
					const std::size_t &batch_size, const std::size_t &stimulus_size,
					const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
					std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
					std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus);
			};
		};
	};
};