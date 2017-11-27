#pragma once

#include <memory>
#include <boost/asio.hpp>
#include "Proxy.h"
#include "Network/Connection.h"
#include "Engine/Dispatcher.h"

class TRN::Engine::Proxy::Handle
{
public:
	std::clock_t start;
	unsigned short number;
	bool quit_broadcasted;
	std::map<int, bool> stopped;
	std::map<unsigned short, std::size_t> frontends;
	std::shared_ptr<TRN::Engine::Communicator> to_workers;
	std::shared_ptr<TRN::Engine::Communicator> to_frontend;
	std::shared_ptr<TRN::Engine::Dispatcher> dispatcher;
	std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> visitor;
};
