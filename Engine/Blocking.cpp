#include "stdafx.h"
#include "Blocking.h"
#include "Executor_impl.h"

TRN::Engine::Blocking::Blocking() : 
	TRN::Engine::Executor()
{

}

void TRN::Engine::Blocking::run()
{
	std::function<void(void)> command;
	while (TRN::Engine::Executor::handle->commands.dequeue(command))
	{
		command();
	}
}


void TRN::Engine::Blocking::run_one()
{
	std::function<void(void)> command;
	if (TRN::Engine::Executor::handle->commands.dequeue(command))
	{
		command();
	}
}

std::shared_ptr<TRN::Engine::Blocking> TRN::Engine::Blocking::create()
{
	return std::make_shared<TRN::Engine::Blocking>();
}
