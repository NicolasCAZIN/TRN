#pragma once

#include "model_global.h"

#include "Core/Decoder.h"

namespace TRN
{
	namespace Model
	{
		namespace Decoder
		{
			namespace Linear
			{
				std::shared_ptr<TRN::Core::Decoder> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, const std::shared_ptr<TRN::Core::Matrix> &cx, const std::shared_ptr<TRN::Core::Matrix> &cy);
			};
			namespace Kernel
			{
				namespace Model
				{
					std::shared_ptr<TRN::Core::Decoder> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver,
						const std::size_t &batch_size, const std::size_t &stimulus_size,
						const std::size_t &rows, const std::size_t &cols,
						const std::pair<float, float> &arena_x, const std::pair<float, float> &arena_y,
						const float &sigma, const float &radius, const float &angle, const float &scale, const unsigned long &seed,
						const std::shared_ptr<TRN::Core::Matrix> &cx,
						const std::shared_ptr<TRN::Core::Matrix> &cy,
						const std::shared_ptr<TRN::Core::Matrix> &K);
				};
				namespace Map
				{
					std::shared_ptr<TRN::Core::Decoder> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver,
						const std::size_t &batch_size, const std::size_t &stimulus_size,
						const std::size_t &rows, const std::size_t &cols,
						const std::pair<float, float> &arena_x, const std::pair<float, float> &arena_y,
						const float &sigma, const float &radius, const float &angle, const float &scale, const unsigned long &seed,
						const std::shared_ptr<TRN::Core::Matrix> &firing_rate_map);
				};
			};
		};
	};
};