#include "stdafx.h"
#include "Receiver_impl.h"

TRN::Engine::Receiver::Receiver() :
	handle(std::make_unique<Handle>())
{
	handle->running = false;
}

TRN::Engine::Receiver::~Receiver()
{
	if (handle->thread.joinable())
		handle->thread.join();

	handle.reset();
}

void TRN::Engine::Receiver::start()
{
	handle->running = true;
	handle->thread = std::thread([&]() 
	{
		initialize();
	
		while (handle->running)
		{
			try
			{
				//std::cout << "receiving" << std::endl;
				receive();
			}
			catch (std::exception &e)
			{
				std::cerr << e.what() << std::endl;
				stop();
			}
		}
		std::cout << "uninitialize" << std::endl;
		uninitialize();

	});
}

void TRN::Engine::Receiver::wait()
{
	// std::cout << __FUNCTION__ << std::endl;
	std::unique_lock<std::mutex> lock(handle->mutex);
	while (handle->running == true)
	{
		//std::cout << "running" << std::endl;
		handle->cond.wait(lock);
	}
//	std::cout << "not running" << std::endl;
}
void TRN::Engine::Receiver::initialize()
{
	// std::cout << __FUNCTION__ << std::endl;
}

void TRN::Engine::Receiver::uninitialize()
{
	// std::cout << __FUNCTION__ << std::endl;
}

bool TRN::Engine::Receiver::is_running()
{
	// std::cout << __FUNCTION__ << std::endl;
	return handle->running;
}
void TRN::Engine::Receiver::stop()
{
	// std::cout << __FUNCTION__ << std::endl;
	std::unique_lock<std::mutex> lock(handle->mutex);
	handle->running = false;
	lock.release();
	handle->cond.notify_one();


}
void TRN::Engine::Receiver::join()
{
	// std::cout << __FUNCTION__ << std::endl;
	if (handle->thread.joinable())
		handle->thread.join();
}