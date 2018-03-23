#pragma once

#include "model_global.h"
#include "Core/Reservoir.h"

namespace TRN
{
	namespace Model
	{
		namespace Reservoir
		{
			namespace WidrowHoff
			{
				std::shared_ptr<TRN::Core::Reservoir> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver,
					const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate,
					const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size, const std::size_t &mini_batch_size);
			};
		};
	};
};