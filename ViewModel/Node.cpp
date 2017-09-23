#include "stdafx.h"
#include "Node.h"
#include "Model/Driver.h"


std::shared_ptr<TRN::Engine::Worker> TRN::ViewModel::Node::Worker::create(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const int &rank, const unsigned int &index)
{
	return TRN::Engine::Worker::create( communicator, rank, TRN::Model::Driver::create(index));
}


std::shared_ptr<TRN::Engine::Proxy> TRN::ViewModel::Node::Proxy::create(const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy, const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor)
{
	return TRN::Engine::Proxy::create(frontend_proxy, proxy_workers, visitor);
}