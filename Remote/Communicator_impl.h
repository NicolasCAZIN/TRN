#pragma once

#include "Communicator.h"
#include "Network/Manager.h"
#include "Network/Messages.h"
#include "Network/Connection.h"
#include "Helper/Queue.h"

class TRN::Remote::Communicator::Handle
{
public :
	std::shared_ptr<TRN::Network::Manager> manager;
	//std::vector<TRN::Network::Processor> processors;
	std::shared_ptr<TRN::Network::Connection> connection;
	//TRN::Helper::Queue<TRN::Network::Data> read;
	//TRN::Helper::Queue<TRN::Network::Data>  write;
	std::mutex write;
	std::recursive_mutex read;
	std::thread transmit, receive;
	int rank;
	std::size_t size;
	TRN::Network::Data received;


	//std::thread reception;
};
