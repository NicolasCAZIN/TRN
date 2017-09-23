#pragma once

#include <memory>
#include <boost/asio.hpp>
#include "Proxy.h"
#include "Network/Connection.h"
#include "Engine/Broker.h"

class TRN::Engine::Proxy::Handle
{
public:
	std::clock_t start;
	std::shared_ptr<TRN::Engine::Communicator> frontend_proxy;
	std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> visitor;
};
