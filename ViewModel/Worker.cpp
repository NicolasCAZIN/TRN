#include "stdafx.h"
#include "Worker.h"
#include "Model/Driver.h"

std::shared_ptr<TRN::Engine::Worker> TRN::ViewModel::Worker::create(const std::shared_ptr<TRN::Engine::Communicator> &communicator, const unsigned int &index)
{
	return TRN::Engine::Worker::create(TRN::Model::Driver::create(index), communicator);
}
