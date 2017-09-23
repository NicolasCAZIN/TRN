#pragma once

#include "Executor.h"
#include "Helper/Queue.h"


class TRN::Engine::Executor::Handle
{
public :
	TRN::Helper::Queue<std::function<void(void)>> commands;

};