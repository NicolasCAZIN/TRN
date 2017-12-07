#include "stdafx.h"

#include "Communicator_impl.h"
#include "Network/Messages.h"

TRN::Remote::Communicator::Communicator(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection, const int &rank, const std::size_t &size) :
	handle(std::make_unique<Handle>())
{
	handle->rank = rank;
	handle->size = size;
	handle->manager = manager;
	
	handle->connection = connection;
	handle->received.tag = -1;
	manager->start(handle->connection);
	/*handle->transmit = std::thread([&]() 
	{
		TRN::Network::Data message;
		while (handle->write.dequeue(message))
		{
			handle->connection->write(message);
		}
	});*/

}

TRN::Remote::Communicator::~Communicator()
{
	// INFORMATION_LOGGER <<   __FUNCTION__ ;



/*	if (handle->transmit.joinable())
		handle->transmit.join();*/
	handle.reset();
}

void TRN::Remote::Communicator::dispose()
{
	handle->manager->stop(handle->connection);
}

int TRN::Remote::Communicator::rank()
{
	return handle->rank;
}

std::size_t TRN::Remote::Communicator::size()
{
	return handle->size;
}

void TRN::Remote::Communicator::send(const int &destination, const TRN::Engine::Tag &tag, const std::string &data)
{
	TRN::Network::Data message;

	message.destination = destination;
	message.tag = tag;
	message.payload = data;

	std::unique_lock<std::mutex> lock(handle->write);
	handle->connection->write(message);
}

std::string TRN::Remote::Communicator::receive(const int &destination, const TRN::Engine::Tag &tag)
{
	std::unique_lock<std::recursive_mutex> lock(handle->read);
	/*if (destination != handle->rank)
		throw std::runtime_error("destination must be != than own rank " + std::to_string(handle->rank));*/
	auto probed = *probe(destination);
	if (probed != tag)
		throw std::invalid_argument("Unexpected tag " + probed);

	auto data = handle->received.payload;
	
	handle->received.payload.clear();
	handle->received.tag = -1;
	handle->received.destination = -1;

	return data;
}

boost::optional<TRN::Engine::Tag> TRN::Remote::Communicator::probe(const int &destination)
{
	std::unique_lock<std::recursive_mutex> lock(handle->read);
	if (handle->received.tag == -1)
	{

		handle->connection->read(handle->received);

	}
	
	if (destination != -1 && destination != handle->rank)
		throw std::runtime_error("destination must be equal to own rank " + std::to_string(handle->rank));
	return boost::optional<TRN::Engine::Tag> (static_cast<TRN::Engine::Tag>(handle->received.tag));
}

std::shared_ptr<TRN::Remote::Communicator> TRN::Remote::Communicator::create(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection, const int &rank, const std::size_t &size)
{
	return std::make_shared<TRN::Remote::Communicator>(manager, connection, rank, size);
}

