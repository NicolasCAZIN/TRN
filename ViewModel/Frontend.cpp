#include "stdafx.h"
#include "Frontend.h"

std::shared_ptr<TRN::Engine::Frontend> TRN::ViewModel::Frontend::create(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const std::shared_ptr<TRN::Engine::Executor> &to_caller)
{
	return TRN::Engine::Frontend::create(communicator, to_caller);
}