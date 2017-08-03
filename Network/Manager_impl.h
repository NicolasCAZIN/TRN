#pragma once

#include "Manager.h"

class TRN::Network::Manager::Handle 
{
public :
	std::set<std::shared_ptr<TRN::Network::Connection>> pool;
	boost::asio::io_service io_service;




};
