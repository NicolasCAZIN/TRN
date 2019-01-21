#include "stdafx.h"
#include "Task_impl.h"
#include "Helper/Logger.h"

TRN::Engine::Task::Task() :
	handle(std::make_unique<Handle>())
{
	handle->running = false;

	TRACE_LOGGER;
}

TRN::Engine::Task::~Task() 
{
	TRACE_LOGGER;
	if (handle->running)
	{
		ERROR_LOGGER << "Task is not terminated while calling the destructor";
		::terminate();
	}
	
	handle.reset();
}

bool TRN::Engine::Task::stop_requested()
{
	if (handle->future_obj.wait_for(std::chrono::milliseconds(0)) == std::future_status::timeout)
		return false;
	return true;
}

void TRN::Engine::Task::start()
{

	TRACE_LOGGER;
	if (handle->thread.get_id() == std::this_thread::get_id())
	{
		throw std::runtime_error("start() must be called from from a different thread");
	}
	
	handle->thread = std::thread([&]()
	{
		handle->running = true;
		initialize();
		while (!stop_requested() && handle->running)
		{
			try
			{
				body();
			}
			catch (std::exception &e)
			{
				ERROR_LOGGER << e.what();
			}
		}

		uninitialize();
	});
	
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

void TRN::Engine::Task::joined()
{
	TRACE_LOGGER;
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
}
void TRN::Engine::Task::cancel()
{
	handle->running = false;
}

void TRN::Engine::Task::stop()
{
	TRACE_LOGGER;
	if (handle->thread.get_id() == std::this_thread::get_id())
	{
		throw std::runtime_error("stop() must be called from from a different thread");
	}
	else
	{
		if (handle->thread.joinable())
		{
			handle->exit_signal.set_value();
			handle->thread.join();
			
			handle->running = false;

			joined();	
		}
		else
		{
			throw std::runtime_error("Thread is not running nor joinable");
		}
	}
}
