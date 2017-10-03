#pragma once

#include "NonBlocking.h"

class TRN::Engine::NonBlocking::Handle
{
public :
	std::thread process;
	bool running;
	std::condition_variable cond;
	std::mutex mutex;
};
