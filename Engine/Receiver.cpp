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
	handle->thread = std::thread([&]() 
	{
		initialize();
		handle->running = true;
		while (is_running())
		{
			try
			{
				receive();
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


void TRN::Engine::Receiver::initialize()
{

}

void TRN::Engine::Receiver::uninitialize()
{

}

bool TRN::Engine::Receiver::is_running()
{
	return handle->running;
}
void TRN::Engine::Receiver::stop()
{
	handle->running = false;
}