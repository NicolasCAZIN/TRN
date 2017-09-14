#pragma once

#include "Acceptor.h"

class TRN::Network::Acceptor::Handle
{
public :
	boost::asio::ip::tcp::acceptor acceptor;
	boost::asio::signal_set signal_set;
	std::shared_ptr<TRN::Network::Manager> manager;
	std::function <void(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection)> on_accept;


	Handle(const std::shared_ptr<TRN::Network::Manager> &manager, const std::function<void(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection)> &on_accept) :
	
		on_accept(on_accept),
		signal_set(manager->get_io_service()),
		acceptor(manager->get_io_service()),
		manager(manager)
	{
		signal_set.add(SIGINT);
		signal_set.add(SIGTERM);
#if defined(SIGQUIT)
		signal_set.add(SIGQUIT);
#endif // defined(SIGQUIT)
	}
};