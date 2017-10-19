#pragma once

#include <memory>
#include <boost/asio.hpp>
#include "Proxy.h"
#include "Network/Connection.h"
#include "Engine/Backend.h"

class TRN::Engine::Proxy::Handle
{
public:
	std::clock_t start;
	std::shared_ptr<TRN::Engine::Backend> backend;
	std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> visitor;
};
