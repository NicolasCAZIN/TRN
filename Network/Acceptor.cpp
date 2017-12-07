#include "stdafx.h"
#include "Acceptor_impl.h"
#include "Helper/Logger.h"

TRN::Network::Acceptor::Acceptor(const std::string &address, const  unsigned short &port,
	const std::function <std::shared_ptr<TRN::Network::Peer> (const std::shared_ptr<TRN::Network::Manager> &manager,
		const std::shared_ptr<TRN::Network::Connection> &connection, const std::function <void(const std::shared_ptr<TRN::Network::Peer> &peer)> &on_terminated)> &on_accept) :
	handle(std::make_unique<Handle>(TRN::Network::Manager::create(), on_accept))
{
	
	boost::asio::ip::tcp::resolver resolver(handle->manager->get_io_service());
	boost::asio::ip::tcp::endpoint endpoint = *resolver.resolve({ address, std::to_string(port) });
	handle->acceptor.open(endpoint.protocol());
	handle->acceptor.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
	handle->acceptor.bind(endpoint);
	handle->acceptor.listen();

	INFORMATION_LOGGER <<   "listening on " << address << ":" << port ;

	accept();
}

TRN::Network::Acceptor::~Acceptor()
{
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	handle.reset();
}

void TRN::Network::Acceptor::run()
{
	handle->manager->get_io_service().run();
}

void TRN::Network::Acceptor::accept()
{
	auto connection = TRN::Network::Connection::create(handle->manager->get_io_service());
	handle->acceptor.async_accept(connection->socket(),
		[this, connection](boost::system::error_code error)
	{
		// Check whether the server was stopped by a signal before this
		// completion handler had a chance to run.
		if (!handle->acceptor.is_open())
		{
			return;
		}

		if (!error)
		{
			auto peer = handle->on_accept(handle->manager, connection, [&](const std::shared_ptr<TRN::Network::Peer> &peer)
			{
				handle->manager->get_io_service().post([=]()
				{
					peer->stop();
					handle->peers.erase(peer);
					INFORMATION_LOGGER <<   "peer destroyed" ;
				});
			});
		
			handle->peers.insert(peer);
			peer->start();
			INFORMATION_LOGGER <<   "peer created" ;


		}

		accept();
	});
}

void TRN::Network::Acceptor::await_stop()
{
	handle->signal_set.async_wait([this](boost::system::error_code /*ec*/, int /*signo*/)
	{
	// The server is stopped by cancelling all outstanding asynchronous
	// operations. Once all operations have finished the io_service::run()
	// call will exit.
		handle->acceptor.close();
		handle->manager->stop();
	});
}




std::shared_ptr<TRN::Network::Acceptor> TRN::Network::Acceptor::create(const std::string &address, const  unsigned short &port,
	const std::function <std::shared_ptr<TRN::Network::Peer>(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection, const std::function <void(const std::shared_ptr<TRN::Network::Peer> &peer)> &on_terminated)> &on_accept)
{
	return std::make_shared<TRN::Network::Acceptor>(address, port, on_accept);
}