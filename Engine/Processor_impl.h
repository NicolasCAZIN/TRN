#pragma once

#include "Processor.h"
#include "Helper/Queue.h"

class TRN::Engine::Processor::Handle
{
public :
	size_t count;
	std::string name;
	std::string host;
	int index;
	int rank;
	float latency;
	clock_t t0;
	clock_t t1;
	std::mutex mutex;
	std::condition_variable cond;
	Status status;
	
};