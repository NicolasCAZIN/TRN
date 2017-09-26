#pragma once

#include "viewmodel_global.h"
#include "Engine/Executor.h"

namespace TRN
{
	namespace ViewModel
	{
		namespace Executor
		{
			namespace Blocking
			{
				std::shared_ptr<TRN::Engine::Executor> VIEWMODEL_EXPORT create();
			};

			namespace NonBlocking
			{
				std::shared_ptr<TRN::Engine::Executor> VIEWMODEL_EXPORT create();
			};
		};
	};
};
