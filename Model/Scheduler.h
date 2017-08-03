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
				std::shared_ptr<TRN::Core::Scheduler> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &epochs);
			};

			namespace Snippets
			{
				std::shared_ptr<TRN::Core::Scheduler> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag);
			};

			namespace Custom
			{
				std::shared_ptr<TRN::Core::Scheduler> MODEL_EXPORT create(const std::shared_ptr<TRN::Backend::Driver> &driver, 
					const  std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)> &request,
					std::function<void(const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)> &reply,
					const std::string &tag
				);
			};
		};
	};
};
