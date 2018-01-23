#pragma once

#include "Manager.h"

class TRN::Network::Manager::Handle 
{
public :
	unsigned short lastid;
	std::set<std::shared_ptr<TRN::Network::Connection>> pool;
	std::map<std::shared_ptr<TRN::Network::Connection>, unsigned short>identified;
	boost::asio::io_service io_service;




};
