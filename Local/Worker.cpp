#include "stdafx.h"
#include "Worker_impl.h"
#include "Model/Driver.h"
#include "Simulator/Basic.h"

TRN::Local::Worker::Worker(const unsigned int &index, const unsigned long &seed) :
	handle(std::make_unique<Handle>())
{
	handle->driver = TRN::Model::Driver::create(index, seed);
}

TRN::Local::Worker::~Worker()
{
	handle.reset();
}


{

}
std::shared_ptr<TRN::Local::Worker> TRN::Local::Worker::create(const unsigned int &index, const unsigned long &seed)
{
	return std::make_shared<TRN::Local::Worker>(index, seed);
}
