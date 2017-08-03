#pragma once

#include "viewmodel_global.h"
#include "Engine/Broker.h"

namespace TRN
{
	namespace ViewModel
	{
		namespace Broker
		{
			std::shared_ptr<TRN::Engine::Broker> VIEWMODEL_EXPORT  create(const std::shared_ptr<TRN::Engine::Communicator> &communicator);
		};
	};
};