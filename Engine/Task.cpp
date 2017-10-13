#include "stdafx.h"
#include "Task_impl.h"

TRN::Engine::Task::Task() :
	handle(std::make_unique<Handle>())
{
	handle->running = false;
}

TRN::Engine::Task::~Task()
{
	if (handle->running)
		throw std::runtime_error("Thread is still running");
	handle.reset();
}

void TRN::Engine::Task::start()
{
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
				std::cerr << e.what() << std::endl;
				stop();
			}
		}
		uninitialize();
	});


}

void TRN::Engine::Task::join()
{
	if (handle->thread.joinable())
		handle->thread.join();
}
void TRN::Engine::Task::initialize()
{
	// std::cout << __FUNCTION__ << std::endl;
}

void TRN::Engine::Task::uninitialize()
{
	// std::cout << __FUNCTION__ << std::endl;
}


void TRN::Engine::Task::stop()
{
	handle->running = false;
}
