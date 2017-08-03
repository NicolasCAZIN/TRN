#pragma once

#include "model_global.h"

#include "Backend/Driver.h"

namespace TRN
{
	namespace Model
	{
		namespace Driver
		{
			std::shared_ptr<TRN::Backend::Driver> MODEL_EXPORT create(const int &index);
			std::list<std::pair<int, std::string>> MODEL_EXPORT enumerate_devices();

		};
	};
};