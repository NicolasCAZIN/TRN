#pragma once

#include "model_global.h"

#include "Core/Scheduler.h"

namespace TRN
{
	namespace Model
	{
		namespace Scheduler
		{
			namespace Tiled
			{
				std::shared_ptr<TRN::Core::Scheduler> MODEL_EXPORT create(const unsigned int &epochs);
			};

			namespace Snippets
			{
				std::shared_ptr<TRN::Core::Scheduler> MODEL_EXPORT create(const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget,
					const float &learn_reverse_rate, const float &generate_reverse_rate,
					const float &learning_rate,
					const float &discount, const std::string &tag);
			};

			namespace Custom
			{
				std::shared_ptr<TRN::Core::Scheduler> MODEL_EXPORT create(const unsigned long &seed,
					const  std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
					std::function<void(const unsigned long long &evaluation_id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply,
					const std::string &tag
				);
			};
		};
	};
};
