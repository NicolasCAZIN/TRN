#pragma once

#include "Broker.h"
#include "Manager.h"
#include "Communicator.h"
#include "Helper/Queue.h"

class TRN::Engine::Broker::Handle 
{
public :


public :
	size_t count;
	std::size_t active;
	std::set<unsigned int> simulations;
	std::shared_ptr<TRN::Engine::Manager> manager;
	std::shared_ptr<TRN::Engine::Communicator> communicator;
	std::map<int, std::shared_ptr<TRN::Engine::Executor>> from_caller;
	std::shared_ptr<TRN::Engine::Executor> to_caller;

	std::recursive_mutex ack;
	std::map<int, std::function<void()>> on_ack_map;
};
