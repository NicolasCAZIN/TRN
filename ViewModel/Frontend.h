#pragma once

#include "viewmodel_global.h"
#include "Engine/Frontend.h"

namespace TRN
{
	namespace ViewModel
	{
		namespace Frontend
		{
			std::shared_ptr<TRN::Engine::Frontend> VIEWMODEL_EXPORT  create(const std::shared_ptr<TRN::Engine::Communicator> &communicator);
		};
	};

};