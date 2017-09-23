#pragma once

#include "NonBlocking.h"

class TRN::Engine::NonBlocking::Handle
{
public :
	std::thread process;
};
