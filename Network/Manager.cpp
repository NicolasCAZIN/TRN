#include "stdafx.h"
#include "Manager_impl.h"
//#include "Helper/Logger.h"

TRN::Network::Manager::Manager() :
	handle(std::make_unique<Handle>())
{

}

TRN::Network::Manager::~Manager()
{
	// std::cout << __FUNCTION__ << std::endl;
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