#pragma once

#include "Backend.h"

class TRN::Engine::Backend::Handle 
{
public :

	std::shared_ptr<TRN::Engine::Communicator> to_frontend;
};