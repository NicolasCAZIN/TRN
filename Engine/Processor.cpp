#include "stdafx.h"
#include "Processor_impl.h"
#include "Helper/Logger.h"

TRN::Engine::Processor::Processor(const int &rank) :

	handle(std::make_unique<Handle>())
{
	TRACE_LOGGER;
	handle->count = 0;
	handle->latency = 0.0f;
	handle->status = TRN::Engine::Processor::Status::Deallocated;
	handle->rank = rank;

}

TRN::Engine::Processor::~Processor()
{
	TRACE_LOGGER;
	handle.reset();
}

int TRN::Engine::Processor::get_rank()
{
	TRACE_LOGGER;
	return handle->rank;
}


void TRN::Engine::Processor::wait(const std::function<bool(const TRN::Engine::Processor::Status &status)> &functor)
{
	TRACE_LOGGER;
	std::unique_lock<std::mutex> lock(handle->mutex);
	while (!functor(handle->status))
		handle->cond.wait(lock);
	lock.unlock();
}

std::ostream &operator << ( std::ostream& stream, const TRN::Engine::Processor::Status &status)
{
	switch (status)
	{
		case TRN::Engine::Processor::Deallocated:
			stream << "Deallocated";
			break;
		case TRN::Engine::Processor::Allocating:
			stream << "Allocating";
			break;
		case TRN::Engine::Processor::Allocated:
			stream << "Allocated";
			break;
		case 	TRN::Engine::Processor::Configuring:
			stream << "Configuring";
			break;

		case 	TRN::Engine::Processor::Configured:
			stream << "Configured";
			break;
		case 	TRN::Engine::Processor::Training:
			stream << "Training";
			break;
		case 	TRN::Engine::Processor::Trained:
			stream << "Trained";
			break;
		case 	TRN::Engine::Processor::Priming:
			stream << "Priming";
			break;
		case 	TRN::Engine::Processor::Primed:
			stream << "Primed";
			break;
		case 	TRN::Engine::Processor::Tested:
			stream << "Tested";
			break;
		case 	TRN::Engine::Processor::Deallocating:
			stream << "Deallocating";
			break;
	}
	return stream;
}



void TRN::Engine::Processor::notify(const TRN::Engine::Processor::Status &status)
{
	TRACE_LOGGER;
	std::unique_lock<std::mutex> lock(handle->mutex);
	auto old_status = handle->status;
	handle->status = status;
	lock.unlock();
	if (old_status != handle->status)
	{
		handle->cond.notify_one();
	
		DEBUG_LOGGER << "rank " << handle->rank << " status changed to " << status ;
	}
}

std::string TRN::Engine::Processor::get_name()
{
	TRACE_LOGGER;
	return handle->name;
}
std::string TRN::Engine::Processor::get_host()
{
	TRACE_LOGGER;
	return handle->host;
}

int TRN::Engine::Processor::get_index()
{
	TRACE_LOGGER;
	return handle->index;
}



void TRN::Engine::Processor::set_t0(const clock_t &t0)
{
	TRACE_LOGGER;
	handle->t0 = t0;
}
void TRN::Engine::Processor::set_t1(const clock_t &t1)
{
	TRACE_LOGGER;
	handle->t1 = t1;
}

 float TRN::Engine::Processor::get_latency()
{
	 TRACE_LOGGER;
	handle->latency = (handle->t1 - handle->t0) / (float)CLOCKS_PER_SEC;
	return handle->latency;
}


 void TRN::Engine::Processor::set_name(const std::string &name)
 {
	 TRACE_LOGGER;
	 handle->name = name;
 }
 void TRN::Engine::Processor::set_host(const std::string &host)
 {
	 TRACE_LOGGER;
	 handle->host = host;
 }
 void TRN::Engine::Processor::set_index(const int &index)
 {
	 TRACE_LOGGER;
	 handle->index = index;
 }

 void TRN::Engine::Processor::allocating()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "allocating() waiting for rank " << handle->rank << " status (" << status << ") == " << TRN::Engine::Processor::Status::Deallocated;
		 return status == TRN::Engine::Processor::Status::Deallocated;
	 });
	 notify(TRN::Engine::Processor::Status::Allocating);
 }

 void TRN::Engine::Processor::allocated()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "allocated() waiting for rank " << handle->rank << " status (" << status << ") == " << TRN::Engine::Processor::Status::Allocating;
		 return status == TRN::Engine::Processor::Status::Allocating;
	 });
	 notify(TRN::Engine::Processor::Status::Allocated);
 }
 void TRN::Engine::Processor::deallocating()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "deallocating() waiting for rank " << handle->rank << " status (" << status << ") == " <<
			 TRN::Engine::Processor::Status::Allocated << "|" <<
			 TRN::Engine::Processor::Status::Configured << "|" <<
			 TRN::Engine::Processor::Status::Trained << "|" <<
			 TRN::Engine::Processor::Status::Tested;
		 return status == TRN::Engine::Processor::Status::Allocated ||
			 status == TRN::Engine::Processor::Status::Configured ||
			 status == TRN::Engine::Processor::Status::Trained ||
			 status == TRN::Engine::Processor::Status::Tested;
	 });
	 notify(TRN::Engine::Processor::Status::Deallocating);
 }

 void TRN::Engine::Processor::deallocated()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "deallocated() waiting for rank " << handle->rank << " status (" << status << ") == " <<
			 TRN::Engine::Processor::Status::Deallocating;
		 return status == TRN::Engine::Processor::Status::Deallocating;
	 });
	 notify(TRN::Engine::Processor::Status::Deallocated);
 }
 void TRN::Engine::Processor::configuring()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "configuring() waiting for rank " << handle->rank << " status (" << status << ") == " <<
			 TRN::Engine::Processor::Status::Configuring << "|" <<
			 TRN::Engine::Processor::Status::Allocated;
		 return status == TRN::Engine::Processor::Status::Configuring || status == TRN::Engine::Processor::Status::Allocated;
	 });
	 notify(TRN::Engine::Processor::Status::Configuring);
 }

 void TRN::Engine::Processor::configured()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "configured() waiting for rank " << handle->rank << " status (" << status << ") == " <<
			 TRN::Engine::Processor::Status::Configuring;
		 return status == TRN::Engine::Processor::Status::Configuring;
	 });
	 notify(TRN::Engine::Processor::Status::Configured);
 }

 void TRN::Engine::Processor::training()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "training() waiting for rank " << handle->rank << " status (" << status << ") == " <<
			 TRN::Engine::Processor::Status::Configured << "|" <<
			 TRN::Engine::Processor::Status::Trained << "|" <<
			 TRN::Engine::Processor::Status::Tested;
		 return status == TRN::Engine::Processor::Status::Configured ||
			 status == TRN::Engine::Processor::Status::Tested || status == TRN::Engine::Processor::Status::Trained;
	 });
	notify(TRN::Engine::Processor::Status::Training);
 }
 void TRN::Engine::Processor::trained()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "trained() waiting for rank " << handle->rank << " status (" << status << ") == " <<
			 TRN::Engine::Processor::Status::Training;
		 return status == TRN::Engine::Processor::Status::Training;
	 });
	 notify(TRN::Engine::Processor::Status::Trained);
 }

 void TRN::Engine::Processor::testing()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "testing() waiting for rank " << handle->rank << " status (" << status << ") == " <<
			 TRN::Engine::Processor::Status::Trained << "|" <<
			 TRN::Engine::Processor::Status::Tested;
		 return status == TRN::Engine::Processor::Status::Trained || status == TRN::Engine::Processor::Status::Tested;
	 });
	 notify(TRN::Engine::Processor::Status::Priming);
 }

 void TRN::Engine::Processor::primed()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "primed() waiting for rank " << handle->rank << " status (" << status << ") == " <<
			 TRN::Engine::Processor::Status::Priming;
		 return status == TRN::Engine::Processor::Status::Priming;
	 });
	 notify(TRN::Engine::Processor::Status::Primed);
 }

 void TRN::Engine::Processor::tested()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "tested() waiting for rank " << handle->rank << " status (" << status << ") == " <<
			 TRN::Engine::Processor::Status::Primed;
		 return status == TRN::Engine::Processor::Status::Primed;
	 });
	 notify(TRN::Engine::Processor::Status::Tested);
 }

 void TRN::Engine::Processor::declare()
 {
	 TRACE_LOGGER;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 DEBUG_LOGGER << "declare() waiting for rank " << handle->rank << " status (" << status << ") == " <<
			 TRN::Engine::Processor::Status::Configured << "|" <<
			 TRN::Engine::Processor::Status::Trained << "|" <<
			 TRN::Engine::Processor::Status::Tested;
		 return status == TRN::Engine::Processor::Status::Configured || TRN::Engine::Processor::Status::Trained || TRN::Engine::Processor::Status::Tested;
	 });
 }


std::shared_ptr<TRN::Engine::Processor> TRN::Engine::Processor::create(const int &rank)
{
	TRACE_LOGGER;
	return std::make_shared<TRN::Engine::Processor>(rank);
}

bool TRN::Engine::operator < (const std::shared_ptr<TRN::Engine::Processor> &left, const std::shared_ptr<TRN::Engine::Processor> &right)
{
	TRACE_LOGGER;
	return left->get_latency() > right->get_latency();
}