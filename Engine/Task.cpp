#include "stdafx.h"
#include "Task_impl.h"
#include "Helper/Logger.h"

TRN::Engine::Task::Task() :
	handle(std::make_unique<Handle>())
{
	TRACE_LOGGER;
	handle->running = false;
}

TRN::Engine::Task::~Task() noexcept(false)
{
	TRACE_LOGGER;
	if (!handle)
		throw std::runtime_error("Handle is already destroyed");
	if (handle->running)
		throw std::runtime_error("Thread is still running");
	handle.reset();
}

void TRN::Engine::Task::start()
{
	TRACE_LOGGER;
	if (handle->running)
		throw std::runtime_error("Thread is already running");
	handle->running = true;
	handle->thread = std::thread([&]() 
	{
	
		initialize();
		while (handle->running)
		{
			try
			{
				body();
			}
			catch (std::exception &e)
			{
				ERROR_LOGGER << e.what() ;
				stop();
			}
		}
		uninitialize();
	});


}

void TRN::Engine::Task::join()
{
	TRACE_LOGGER;
	if (handle->thread.joinable())
		handle->thread.join();

}
void TRN::Engine::Task::initialize()
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
}

void TRN::Engine::Task::uninitialize()
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
}


void TRN::Engine::Task::stop()
{
	TRACE_LOGGER;
	handle->running = false;
}
