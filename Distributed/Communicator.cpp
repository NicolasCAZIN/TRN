#include "stdafx.h"
#include "Communicator_impl.h"

TRN::Distributed::Communicator::Communicator(int argc, char *argv[]) :
	handle(std::make_unique<Handle>(argc, argv))

{
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	if (size() <= 1)
		throw std::runtime_error("At least, one MPI worker is required");


}
void TRN::Distributed::Communicator::dispose()
{
	handle->world.abort(0);
}
TRN::Distributed::Communicator::~Communicator()
{
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	handle.reset();
}

int TRN::Distributed::Communicator::rank()
{
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	return handle->world.rank();
}

std::size_t TRN::Distributed::Communicator::size()
{

	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	return handle->world.size();
}

void TRN::Distributed::Communicator::send(const int &destination, const TRN::Engine::Tag &tag, const std::string &data)
{
	handle->world.send(destination, tag, data);
}
std::string TRN::Distributed::Communicator::receive(const int &destination, const TRN::Engine::Tag &tag)
{

	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	std::string data;

	handle->world.recv(boost::mpi::any_source, tag, data);

	return data;
}

boost::optional<TRN::Engine::Tag> TRN::Distributed::Communicator::probe(const int &destination)
{
	// INFORMATION_LOGGER <<   __FUNCTION__ ;
	auto status = handle->world.probe(boost::mpi::any_source, boost::mpi::any_tag);
	return boost::optional<TRN::Engine::Tag>(static_cast<TRN::Engine::Tag>(status.tag()));
}

std::shared_ptr<TRN::Distributed::Communicator> TRN::Distributed::Communicator::create(int argc, char *argv[])
{
	return std::make_shared<TRN::Distributed::Communicator>(argc, argv);
}

