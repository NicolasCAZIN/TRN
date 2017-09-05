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
				std::shared_ptr<TRN::Core::Mutator> MODEL_EXPORT create();
			};

			namespace Reverse
			{
				std::shared_ptr<TRN::Core::Mutator> MODEL_EXPORT create(const float &rate, const std::size_t &size);
			};

			namespace Punch
			{
				std::shared_ptr<TRN::Core::Mutator> MODEL_EXPORT create(const float &rate, const std::size_t &size, const std::size_t &number);
			};

			namespace Custom
			{
				std::shared_ptr<TRN::Core::Mutator> MODEL_EXPORT create(const std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &request, std::function<void(const std::vector<int> &offsets, const std::vector<int> &durations)> &reply);
			};
		};
	};
};