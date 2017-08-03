#pragma once

#include "viewmodel_global.h"
#include "Engine/Worker.h"

namespace TRN
{
	namespace ViewModel
	{
		namespace Worker
		{
			std::shared_ptr<TRN::Engine::Worker> VIEWMODEL_EXPORT  create(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const unsigned int &index);
		};

	};

};