#pragma once

#include "viewmodel_global.h"
#include "Engine/Worker.h"
#include "Engine/Proxy.h"
#include "Engine/Broker.h"

namespace TRN
{
	namespace ViewModel
	{
		namespace Node
		{
			namespace Backend
			{
				std::shared_ptr<TRN::Engine::Worker> VIEWMODEL_EXPORT  create(const std::shared_ptr<TRN::Engine::Communicator> &communicator,const int &rank,  const unsigned int &index);
			}

			namespace Proxy
			{
				std::shared_ptr<TRN::Engine::Proxy> VIEWMODEL_EXPORT  create(const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy, 
					const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers, const std::shared_ptr<TRN::Engine::Dispatcher> &dispatcher, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor, const unsigned short &simulation_id);
			}
		};

	};

};