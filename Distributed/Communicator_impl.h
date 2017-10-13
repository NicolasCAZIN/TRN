#pragma once

#include "Communicator.h"
#include "Helper/Queue.h"

class TRN::Distributed::Communicator::Handle
{
public:
	boost::mpi::environment env;
	boost::mpi::communicator world;
public :
	Handle(int argc, char *argv[]) : env(argc, argv, boost::mpi::threading::level::multiple, false)
	{
		if (!env.initialized())
		{
			throw std::runtime_error("MPI environment not initialized");
		}
	}

};
