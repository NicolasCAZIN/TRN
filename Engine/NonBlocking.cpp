#include "stdafx.h"
#include "NonBlocking_impl.h"
#include "Executor_impl.h"

TRN::Engine::NonBlocking::NonBlocking() :
	TRN::Engine::Executor(),
	handle(std::make_unique<Handle>())
{
	handle->joined = false;
	handle->process = std::thread([&]()
	{
	
		std::function<void(void)> command;
		while (TRN::Engine::Executor::handle->commands.dequeue(command))
		{
			command();
		}
		//std::cout << "stopped" << std::endl;
	});
}

TRN::Engine::NonBlocking::~NonBlocking()
{
	join();
	handle.reset();
}
void TRN::Engine::NonBlocking::join()
{
	if (!handle->joined)
	{
		if (handle->process.joinable())
			handle->process.join();
		handle->joined = true;
	}
}


void TRN::Engine::NonBlocking::run()
{
	join();
}

void TRN::Engine::NonBlocking::run_one()
{
}

std::shared_ptr<TRN::Engine::NonBlocking> TRN::Engine::NonBlocking::create()
{
	return std::make_shared<TRN::Engine::NonBlocking>();
}