#pragma once

#include "presenter_global.h"
#include "Engine/Communicator.h"
#include "Remote/Communicator.h"
namespace TRN
{
	namespace ViewModel
	{
		namespace Communicator
		{
			namespace Local
			{
				std::shared_ptr<TRN::Engine::Communicator> PRESENTER_EXPORT  create(const std::vector<unsigned int> &indexes);
			};

			namespace Remote
			{
			std::shared_ptr<TRN::Engine::Communicator> PRESENTER_EXPORT  create(const std::string &host, const unsigned short &port);
			};

			
			namespace Distributed
			{
				std::shared_ptr<TRN::Engine::Communicator> PRESENTER_EXPORT  create(int argc, char *argv[]);
			};
		};
	};

};