#include "stdafx.h"
#include "Broker.h"

std::shared_ptr<TRN::Engine::Broker> TRN::ViewModel::Broker::create(const std::shared_ptr<TRN::Engine::Communicator> &communicator)
{
	return TRN::Engine::Broker::create(communicator);
}


