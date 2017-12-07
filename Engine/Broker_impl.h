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
	bool completed;
	std::map<int, std::set<unsigned int>> cached;
	std::set<unsigned long long> simulations;
	std::shared_ptr<TRN::Engine::Manager> manager;
	std::shared_ptr<TRN::Engine::Communicator> communicator;
	std::map<unsigned long long, std::shared_ptr<TRN::Engine::Executor>> from_caller;
	std::shared_ptr<TRN::Engine::Executor> to_caller;
	std::mutex counter;
	std::mutex mutex;
	std::mutex cache_mutex;
	std::map<int, std::string> rank_host;
	std::map<std::string, std::vector<int>> host_ranks;

	/*std::recursive_mutex ack;
	std::map<std::size_t, std::function<void()>> on_ack_map;*/
};
