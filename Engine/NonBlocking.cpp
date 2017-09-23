#include "stdafx.h"
#include "NonBlocking_impl.h"
#include "Executor_impl.h"

TRN::Engine::NonBlocking::NonBlocking() :
	TRN::Engine::Executor(),
	handle(std::make_unique<Handle>())
{
	handle->process = std::thread([&]()
	{
		std::function<void(void)> command;
		while (TRN::Engine::Executor::handle->commands.dequeue(command))
		{
			command();
		}
	});
}

TRN::Engine::NonBlocking::~NonBlocking()
{
	TRN::Engine::Executor::handle->commands.invalidate();
	if (handle->process.joinable())
		handle->process.join();
	handle.reset();
}

void TRN::Engine::NonBlocking::run()
{
	if (handle->process.joinable())
		handle->process.join();
}

std::shared_ptr<TRN::Engine::NonBlocking> TRN::Engine::NonBlocking::create()
{
	return std::make_shared<TRN::Engine::NonBlocking>();
}