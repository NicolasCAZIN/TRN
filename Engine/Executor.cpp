#include "stdafx.h"
#include "Executor_impl.h"

TRN::Engine::Executor::Executor() :
	handle(std::make_unique<Handle>())
{
}

TRN::Engine::Executor::~Executor()
{
	handle.reset();
}
void TRN::Engine::Executor::terminate()
{
	handle->commands.invalidate();
	join();
}

void TRN::Engine::Executor::body()
{
	std::function<void(void)> command;
	if (TRN::Engine::Executor::handle->commands.dequeue(command))
	{
		command();
	}
	else
	{
		stop();
	}
}


void TRN::Engine::Executor::post(const std::function<void(void)> &command)
{
	handle->commands.enqueue(command);
}

std::shared_ptr<TRN::Engine::Executor> TRN::Engine::Executor::create()
{
	return std::make_shared<TRN::Engine::Executor>();
}