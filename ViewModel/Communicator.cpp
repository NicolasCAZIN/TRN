#include "stdafx.h"
#include "Communicator.h"
#include "Local/Communicator.h"
#include "Remote/Communicator.h"
#include "Distributed/Communicator.h"


#include "Model/Driver.h"
#include "Engine/Worker.h"
#include "Node.h"
#include "Helper/Logger.h"
/*#include "Remote/Backend.h"
#include "Distributed/Backend.h"
*/
std::shared_ptr<TRN::Engine::Communicator> TRN::ViewModel::Communicator::Local::create(const std::vector<unsigned int> &indexes)
{
	std::vector<unsigned int> local_indexes;

	if (indexes.empty())
	{
		auto devices = TRN::Model::Driver::enumerate_devices();
		for (auto device : devices)
		{
			local_indexes.push_back(device.first);
		}
	}
	else
	{
		local_indexes = indexes;
	}

	auto communicator = TRN::Local::Communicator::create(local_indexes.size() + 1);
	for (int k = 0; k < local_indexes.size(); k++)
	{
		auto index = local_indexes[k];
		auto worker = TRN::ViewModel::Node::Backend::create(communicator, k + 1, index);
		worker->start();
		communicator->append(worker);
	}

	return communicator;
}

std::shared_ptr<TRN::Engine::Communicator> TRN::ViewModel::Communicator::Distributed::create(int argc, char *argv[])
{

	return TRN::Distributed::Communicator::create(argc, argv);
}





std::shared_ptr<TRN::Engine::Communicator> TRN::ViewModel::Communicator::Remote::create(const std::string &host, const unsigned short &port)
{
	auto manager = TRN::Network::Manager::create();
	auto connection = TRN::Network::Connection::create(manager->get_io_service());
	boost::asio::ip::tcp::resolver resolver(manager->get_io_service());
	boost::asio::ip::tcp::resolver::iterator endpoint_iterator = resolver.resolve({ host, std::to_string(port) });
	boost::asio::ip::tcp::resolver::iterator end;
	boost::system::error_code error = boost::asio::error::would_block;

	for (int attempt = 0; attempt < 10; attempt++)
	{
		boost::asio::connect(connection->socket(), endpoint_iterator, error);
		if (!error)
			break;
		else
			Sleep(1000);
	}
	if (error)
		throw boost::system::system_error(error);
	//do connection->socket().get_io_service().run_one(); while (error == boost::asio::error::would_block);

	assert(connection->socket().is_open());
	INFORMATION_LOGGER <<   "API local " << connection->socket().local_endpoint().address().to_string() << ":" << connection->socket().local_endpoint().port() << " remote " << connection->socket().remote_endpoint().address().to_string() << ":" << connection->socket().remote_endpoint().port() ;

	std::size_t size = 0;
	connection->read(size);

	return TRN::Remote::Communicator::create(manager, connection, 0, size);
}

