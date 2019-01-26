#pragma once

#include "presenter_global.h"
#include "Engine/Frontend.h"

namespace TRN
{
	namespace ViewModel
	{
		namespace Frontend
		{
			std::shared_ptr<TRN::Engine::Frontend> PRESENTER_EXPORT  create(const std::shared_ptr<TRN::Engine::Communicator> &communicator);
		};
	};

};