#include "stdafx.h"
#include "Manager_impl.h"
#include "Helper/Logger.h"

TRN::Network::Manager::Manager() :
	handle(std::make_unique<Handle>())
{

}

TRN::Network::Manager::~Manager()
{
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	stop();
	handle.reset();
}


boost::asio::io_service &TRN::Network::Manager::get_io_service()
{
	return handle->io_service;
}


void TRN::Network::Manager::stop()
{
//	LOG_INFO("Stopping all connections");
	for (auto connection : handle->pool)
	{
		stop(connection);
	}
}
void TRN::Network::Manager::start(const std::shared_ptr<TRN::Network::Connection> connection)
{
//	LOG_INFO("Starting connection");
	if (handle->pool.find(connection) != handle->pool.end())
	{
		throw std::runtime_error("Connection is already managed");
	}
	

	auto it = std::max_element(handle->pool.begin(), handle->pool.end(), [](const std::shared_ptr<TRN::Network::Connection> &lhs, const std::shared_ptr<TRN::Network::Connection> &rhs)
	{
		return lhs->get_id() < rhs->get_id();
	});

	unsigned short id = 0;

	if (it == handle->pool.end())
	{
		id = 1;
	}
	else
	{
		id = (*it)->get_id() + 1;
	}
	INFORMATION_LOGGER <<   "Connectionid = " << id;
	connection->set_id(id);
	handle->pool.insert(connection);

	//connection->start();
}
void TRN::Network::Manager::stop(const std::shared_ptr<TRN::Network::Connection> connection)
{
//	LOG_INFO("Stopping connection");
	handle->pool.erase(connection);
	//connection->stop();
}
std::shared_ptr<TRN::Network::Manager> TRN::Network::Manager::create()
{
	return std::make_shared<TRN::Network::Manager>();
}