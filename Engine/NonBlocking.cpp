#include "stdafx.h"
#include "NonBlocking_impl.h"
#include "Executor_impl.h"

TRN::Engine::NonBlocking::NonBlocking() :
	TRN::Engine::Executor(),
	handle(std::make_unique<Handle>())
{
	handle->running = true;
	handle->process = std::thread([&]()
	{
	
		std::function<void(void)> command;
		while (TRN::Engine::Executor::handle->commands.dequeue(command))
		{
			command();
		}
		std::unique_lock<std::mutex> lock(handle->mutex);
		handle->running = false;
		handle->cond.notify_one();
	});
}

TRN::Engine::NonBlocking::~NonBlocking()
{

	handle.reset();
}
void TRN::Engine::NonBlocking::join()
{
	if (handle->running)
	{
		if (handle->process.joinable())
			handle->process.join();
	}
}


void TRN::Engine::NonBlocking::run()
{
	std::unique_lock<std::mutex> lock(handle->mutex);

	while (handle->running)
		handle->cond.wait(lock);
}

void TRN::Engine::NonBlocking::run_one()
{
}

std::shared_ptr<TRN::Engine::NonBlocking> TRN::Engine::NonBlocking::create()
{
	return std::make_shared<TRN::Engine::NonBlocking>();
}