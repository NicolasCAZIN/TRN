#pragma once

#include "Receiver.h"

class TRN::Engine::Receiver::Handle
{
public :
	bool running;
	std::thread thread;
	std::mutex mutex;
	std::condition_variable cond;
};
