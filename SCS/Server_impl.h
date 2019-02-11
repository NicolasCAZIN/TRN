#pragma once

#include "Server.h"

class Server::Handle
{
	zmq::context_t context;
	zmq::socket_t socket;
};
