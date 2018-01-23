#pragma once

#include "model_global.h"
#include "Core/Mutator.h"

namespace TRN
{
	namespace Model
	{
		namespace Mutator
		{
			namespace Shuffle
			{
				std::shared_ptr<TRN::Core::Mutator> MODEL_EXPORT create(const unsigned long &seed);
			};

			namespace Reverse
			{
				std::shared_ptr<TRN::Core::Mutator> MODEL_EXPORT create(const unsigned long &seed, const float &rate, const std::size_t &size);
			};

			namespace Punch
			{
				std::shared_ptr<TRN::Core::Mutator> MODEL_EXPORT create(const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &counter);
			};

			namespace Custom
			{
				std::shared_ptr<TRN::Core::Mutator> MODEL_EXPORT create(const unsigned long &seed, const std::function<void(const unsigned long long &evaluation_id, const unsigned long &seed,  const std::vector<int> &offsets, const std::vector<int> &durations)> &request, std::function<void(const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply);
			};
		};
	};
};