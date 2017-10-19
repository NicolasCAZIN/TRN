#pragma once

#include "Acceptor.h"


class TRN::Network::Acceptor::Handle
{
public :
	boost::asio::ip::tcp::acceptor acceptor;
	boost::asio::signal_set signal_set;
	std::shared_ptr<TRN::Network::Manager> manager;
	std::set<std::shared_ptr<TRN::Network::Peer>> peers;
	std::function <std::shared_ptr<TRN::Network::Peer>(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection, const std::function <void(const std::shared_ptr<TRN::Network::Peer> &peer)> &on_terminated)> on_accept;


	Handle(const std::shared_ptr<TRN::Network::Manager> &manager, const std::function<std::shared_ptr<TRN::Network::Peer>(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection, const std::function <void(const std::shared_ptr<TRN::Network::Peer> &peer)> &on_terminated)> &on_accept) :
	
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