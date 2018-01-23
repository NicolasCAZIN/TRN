#pragma once

#include "model_global.h"

#include "Core/Loop.h"

namespace TRN
{
	namespace Model
	{
		namespace Loop
		{
			namespace Copy
			{
				std::shared_ptr<TRN::Core::Loop> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size);
			};
			namespace SpatialFilter
			{
				std::shared_ptr<TRN::Core::Loop> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
					const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
					std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
					const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
					std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
					const std::size_t &rows, const std::size_t &cols,
					const std::pair<float, float> &x, const std::pair<float, float> &y,
					const std::shared_ptr<TRN::Core::Matrix> &firing_rate_map,
					const float &sigma,
					const float &radius,
					const float &scale,
					const std::string &tag);
			};
			namespace Custom
			{
				std::shared_ptr<TRN::Core::Loop> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
					const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
					std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply);
			};
		};
	};
};