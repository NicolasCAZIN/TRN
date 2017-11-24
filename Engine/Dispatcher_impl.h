#pragma once

#include "Dispatcher.h"

class TRN::Engine::Dispatcher::Handle 
{
public :
	std::map<unsigned short, std::shared_ptr<TRN::Engine::Communicator>> to_frontend;

};