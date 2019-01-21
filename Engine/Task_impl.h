#pragma once

#include "Task.h"

class TRN::Engine::Task::Handle
{
public :
	std::promise<void> exit_signal;
	std::future<void> future_obj;
	std::thread thread;
	bool running;

	Handle() : future_obj(exit_signal.get_future())
	{

	}

	Handle(Handle && obj) : exit_signal(std::move(obj.exit_signal)), future_obj(std::move(obj.future_obj))
	{

	}



};
