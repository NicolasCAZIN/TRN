#pragma once

#include "Receiver.h"

class TRN::Engine::Receiver::Handle
{
public :
	bool running;
	std::thread thread;
};
