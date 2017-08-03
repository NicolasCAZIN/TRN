#include "stdafx.h"
#include "Acceptor_impl.h"


TRN::Network::Acceptor::Acceptor(const std::string &host, const std::string &port,
	const std::function <const std::shared_ptr<TRN::Network::Connection>(boost::asio::ip::tcp::socket, const std::shared_ptr<TRN::Network::Manager>)> &create) :
	handle(std::make_unique<Handle>(TRN::Network::Manager::create(), create))
{

	boost::asio::ip::tcp::resolver resolver(*handle->manager);
	boost::asio::ip::tcp::endpoint endpoint = *resolver.resolve({ host, port });
	handle->acceptor.open(endpoint.protocol());
	handle->acceptor.set_option(boost::asio::ip::tcp::acceptor::reuse_address(true));
	handle->acceptor.bind(endpoint);
	handle->acceptor.listen();

	std::cout << "listening on " << host << ":" << port << std::endl;

	accept();
}

TRN::Network::Acceptor::~Acceptor()
{
	handle.reset();
}

void TRN::Network::Acceptor::run()
{
	handle->manager->run();
}

void TRN::Network::Acceptor::accept()
{
	handle->acceptor.async_accept(handle->socket,
		[this](boost::system::error_code error)
	{
		// Check whether the server was stopped by a signal before this
		// completion handler had a chance to run.
		if (!handle->acceptor.is_open())
		{
			return;
		}

		if (!error)
		{
			handle->manager->start(handle->create(std::move(handle->socket), handle->manager));
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




std::shared_ptr<TRN::Network::Acceptor> TRN::Network::Acceptor::create(const std::string &address, const std::string &port,
	const std::function <const std::shared_ptr<TRN::Network::Connection>(boost::asio::ip::tcp::socket, const std::shared_ptr<TRN::Network::Manager>)> &create)
{
	return std::make_shared<TRN::Network::Acceptor>(address, port, create);
}