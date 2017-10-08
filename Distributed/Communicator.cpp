#include "stdafx.h"
#include "Communicator_impl.h"

TRN::Distributed::Communicator::Communicator(int argc, char *argv[]) :
	handle(std::make_unique<Handle>(argc, argv))

{
	// std::cout << __FUNCTION__ << std::endl;
	if (size() <= 1)
		throw std::runtime_error("At least, one MPI worker is required");
}

TRN::Distributed::Communicator::~Communicator()
{
	// std::cout << __FUNCTION__ << std::endl;
	handle.reset();
}

int TRN::Distributed::Communicator::rank()
{
	// std::cout << __FUNCTION__ << std::endl;
	return handle->world.rank();
}

std::size_t TRN::Distributed::Communicator::size()
{
	// std::cout << __FUNCTION__ << std::endl;
	return handle->world.size();
}

void TRN::Distributed::Communicator::send(const int &destination, const TRN::Engine::Tag &tag, const std::string &data)
{
	// std::cout << __FUNCTION__ <<  std::endl;
	handle->world.send(destination, tag, data);
}
std::string TRN::Distributed::Communicator::receive(const int &destination, const TRN::Engine::Tag &tag)
{
	// std::cout << __FUNCTION__ << std::endl;
	std::string data;

	handle->world.recv(boost::mpi::any_source, tag, data);

	return data;
}

TRN::Engine::Tag TRN::Distributed::Communicator::probe(const int &destination)
{
	// std::cout << __FUNCTION__ << std::endl;
	auto status = handle->world.probe(boost::mpi::any_source, boost::mpi::any_tag);

	return (TRN::Engine::Tag)status.tag();
}

std::shared_ptr<TRN::Distributed::Communicator> TRN::Distributed::Communicator::create(int argc, char *argv[])
{
	return std::make_shared<TRN::Distributed::Communicator>(argc, argv);
}

