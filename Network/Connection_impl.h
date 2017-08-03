#pragma once

#include "Connection.h"
#include "Manager.h"
class TRN::Network::Connection::Handle
{
	public :
		std::size_t expected;
		boost::asio::streambuf rx;
	
		boost::asio::deadline_timer deadline_timer;
		boost::asio::ip::tcp::socket socket;
		std::shared_ptr<TRN::Network::Manager> manager;

public :
	Handle(const std::shared_ptr<TRN::Network::Manager> manager, boost::asio::ip::tcp::socket socket) :
		socket(std::move(socket)),
		deadline_timer(*manager),
		manager(manager)
	{

	}
	Handle(const std::shared_ptr<TRN::Network::Manager> manager) :
		socket(*manager),
		deadline_timer(*manager),
		manager(manager)
	{

	}

};