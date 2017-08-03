#pragma once

#include "viewmodel_global.h"
#include "Engine/Communicator.h"

namespace TRN
{
	namespace ViewModel
	{
		namespace Communicator
		{
			namespace Local
			{
				std::shared_ptr<TRN::Engine::Communicator> VIEWMODEL_EXPORT  create(const std::list<unsigned int> &indexes);
			};

			/*namespace Remote
			{
			std::shared_ptr<TRN::Engine::Backend> VIEWMODEL_EXPORT  create(const std::string &host, const unsigned short &port);
			};

			*/
			namespace Distributed
			{
				std::shared_ptr<TRN::Engine::Communicator> VIEWMODEL_EXPORT  create(int argc, char *argv[]);
			};
		};
	};

};