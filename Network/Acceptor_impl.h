#pragma once

#include "Acceptor.h"

class TRN::Network::Acceptor::Handle
{
public :
	boost::asio::ip::tcp::acceptor acceptor;
	boost::asio::signal_set signal_set;
	boost::asio::ip::tcp::socket socket;
	std::shared_ptr<TRN::Network::Manager> manager;
	std::function <const std::shared_ptr<TRN::Network::Connection>(boost::asio::ip::tcp::socket, const std::shared_ptr<TRN::Network::Manager>)> create;


	Handle(const std::shared_ptr<TRN::Network::Manager> manager, const std::function <const std::shared_ptr<TRN::Network::Connection>(boost::asio::ip::tcp::socket, const std::shared_ptr<TRN::Network::Manager>)> &create) :
	
		create(create),
		signal_set(*manager),
		acceptor(*manager),
		socket(*manager),
		manager(manager)
	{
		signal_set.add(SIGINT);
		signal_set.add(SIGTERM);
#if defined(SIGQUIT)
		signal_set.add(SIGQUIT);
#endif // defined(SIGQUIT)
	}
};