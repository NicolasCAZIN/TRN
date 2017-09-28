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
	handle->commands.enqueue([&]() 
	{
		handle->commands.invalidate(); 
		//std::cout << "invalidated" << std::endl;
	});
	//join();
}

void TRN::Engine::Executor::post(const std::function<void(void)> &command)
{
	handle->commands.enqueue(command);
}