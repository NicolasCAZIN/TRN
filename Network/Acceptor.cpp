#include "stdafx.h"
#include "Acceptor_impl.h"


TRN::Network::Acceptor::Acceptor(const std::string &host, const unsigned short &port,
	const std::function <void(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection)> &on_accept) :
	handle(std::make_unique<Handle>(TRN::Network::Manager::create(), on_accept))
{

	boost::asio::ip::tcp::resolver resolver(handle->manager->get_io_service());
	boost::asio::ip::tcp::endpoint endpoint = *resolver.resolve({ host, std::to_string(port) });
	handle->acceptor.open(endpoint.protocol());
	handle->acceptor.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
	handle->acceptor.bind(endpoint);
	handle->acceptor.listen();

	std::cout << "listening on " << host << ":" << port << std::endl;

	accept();
}

TRN::Network::Acceptor::~Acceptor()
{
	std::cout << __FUNCTION__ << std::endl;
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
			handle->on_accept(handle->manager, connection);
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
	const std::function <void(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection)> &on_accept)
{
	return std::make_shared<TRN::Network::Acceptor>(address, port, on_accept);
}