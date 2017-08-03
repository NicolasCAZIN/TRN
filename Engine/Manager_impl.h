#pragma once

#include "Manager.h"
#include "Helper/Queue.h"
class TRN::Engine::Manager::Handle
{
public :
	std::vector<std::shared_ptr<TRN::Engine::Processor>> processors;
	std::map<int, std::shared_ptr<TRN::Engine::Processor>> associated;
	std::priority_queue<std::shared_ptr<TRN::Engine::Processor>> available;
	std::condition_variable condition;
	std::mutex mutex;

	TRN::Helper::Queue<unsigned int> to_deallocate;

	std::thread deallocator;
};


