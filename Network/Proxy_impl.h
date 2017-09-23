#pragma once

#include "Proxy.h"

class TRN::Network::Proxy::Handle
{
public :

	std::thread client_workers;
	std::thread workers_client;
};