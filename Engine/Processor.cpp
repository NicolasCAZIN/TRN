#include "stdafx.h"
#include "Processor_impl.h"


TRN::Engine::Processor::Processor(const int &rank) :

	handle(std::make_unique<Handle>())
{
	handle->count = 0;
	handle->latency = 0.0f;
	handle->status = TRN::Engine::Processor::Status::Deallocated;
	handle->rank = rank;

}

TRN::Engine::Processor::~Processor()
{
	handle.reset();
}

int TRN::Engine::Processor::get_rank()
{
	return handle->rank;
}


void TRN::Engine::Processor::wait(const std::function<bool(const TRN::Engine::Processor::Status &status)> &functor)
{
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
		/*case TRN::Engine::Processor::Ready:
			stream << "Ready";
			break;*/
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
	std::unique_lock<std::mutex> lock(handle->mutex);
	auto old_status = handle->status;
	handle->status = status;
	lock.unlock();
	if (old_status != handle->status)
	{
		handle->cond.notify_one();
	
		

		//PrintThread{} << "rank " << handle->rank << " status changed to " << status << std::endl;
	}
}

std::string TRN::Engine::Processor::get_name()
{
	return handle->name;
}
std::string TRN::Engine::Processor::get_host()
{
	return handle->host;
}

int TRN::Engine::Processor::get_index()
{
	return handle->index;
}



void TRN::Engine::Processor::set_t0(const clock_t &t0)
{
	handle->t0 = t0;
}
void TRN::Engine::Processor::set_t1(const clock_t &t1)
{
	handle->t1 = t1;
}

 float TRN::Engine::Processor::get_latency()
{
	handle->latency = (handle->t1 - handle->t0) / (float)CLOCKS_PER_SEC;
	return handle->latency;
}


 void TRN::Engine::Processor::set_name(const std::string &name)
 {
	 handle->name = name;
 }
 void TRN::Engine::Processor::set_host(const std::string &host)
 {
	 handle->host = host;
 }
 void TRN::Engine::Processor::set_index(const int &index)
 {
	 handle->index = index;
 }

 void TRN::Engine::Processor::allocating()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 //PrintThread{} << "waiting for rank " << handle->rank << " status " << status << " == " << TRN::Engine::Processor::Status::Deallocated << std::endl;
		 return status == TRN::Engine::Processor::Status::Deallocated;
	 });
	 notify(TRN::Engine::Processor::Status::Allocating);
	 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }

 void TRN::Engine::Processor::allocated()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 //PrintThread{} << "waiting for rank " << handle->rank << " status " << status << " == " << TRN::Engine::Processor::Status::Allocating << std::endl;
		 return status == TRN::Engine::Processor::Status::Allocating;
	 });
	 notify(TRN::Engine::Processor::Status::Allocated);
	 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }
 void TRN::Engine::Processor::deallocating()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 /*PrintThread{} << "waiting for rank " << handle->rank <<
			 " status " << status << " == " << TRN::Engine::Processor::Status::Allocated <<
			 " || " <<
			 " status " << status << " == " << TRN::Engine::Processor::Status::Configured <<
			 " || " <<
			 " status " << status << " == " << TRN::Engine::Processor::Status::Trained <<
			 " || " <<
			 " status " << status << " == " << TRN::Engine::Processor::Status::Tested <<
			 std::endl;*/
		 return status == TRN::Engine::Processor::Status::Allocated ||
			 status == TRN::Engine::Processor::Status::Configured ||
			 status == TRN::Engine::Processor::Status::Trained ||
			 status == TRN::Engine::Processor::Status::Tested;
	 });
	 notify(TRN::Engine::Processor::Status::Deallocating);
	 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }

 void TRN::Engine::Processor::deallocated()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 //PrintThread{} << "waiting for rank " << handle->rank << " status " << status << " == " << TRN::Engine::Processor::Status::Deallocating << std::endl;
		 return status == TRN::Engine::Processor::Status::Deallocating;
	 });
	 notify(TRN::Engine::Processor::Status::Deallocated);
	 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }
 void TRN::Engine::Processor::configuring()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 /*PrintThread{} << "waiting for rank " << handle->rank << 
			 " status " << status << " == " << TRN::Engine::Processor::Status::Configuring <<
			 " || " << 
			 " status " << status << " == " << TRN::Engine::Processor::Status::Allocated <<  std::endl;*/
		 return status == TRN::Engine::Processor::Status::Configuring || status == TRN::Engine::Processor::Status::Allocated;
	 });
	 notify(TRN::Engine::Processor::Status::Configuring);
	 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }

 void TRN::Engine::Processor::configured()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		// PrintThread{} << "waiting for rank " << handle->rank << " status " << status << " == " << TRN::Engine::Processor::Status::Configuring  << std::endl;
		 return status == TRN::Engine::Processor::Status::Configuring;
	 });
	 notify(TRN::Engine::Processor::Status::Configured);
	 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }

 void TRN::Engine::Processor::training()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
	//	 PrintThread{} << "waiting for rank " << handle->rank <<
/*			 " status " << status << " == " << TRN::Engine::Processor::Status::Ready <<
			 " || " <<
			 " status " << status << " == " << TRN::Engine::Processor::Status::Configured <<
			 " || " <<
			 " status " << status << " == " << TRN::Engine::Processor::Status::Tested << std::endl;*/
		 return status == TRN::Engine::Processor::Status::Configured ||
			 status == TRN::Engine::Processor::Status::Tested || status == TRN::Engine::Processor::Status::Trained;
	 });
	notify(TRN::Engine::Processor::Status::Training);

	//PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }
 void TRN::Engine::Processor::trained()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		// PrintThread{} << "waiting for rank " << handle->rank << " status " << status << " == " << TRN::Engine::Processor::Status::Training << std::endl;
		 return status == TRN::Engine::Processor::Status::Training;
	 });
	 notify(TRN::Engine::Processor::Status::Trained);
	 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }

 void TRN::Engine::Processor::testing()
 {
	// PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		// PrintThread{} << "waiting for rank " << handle->rank << " status " << status << " == " << TRN::Engine::Processor::Status::Trained << std::endl;
		 return status == TRN::Engine::Processor::Status::Trained || status == TRN::Engine::Processor::Status::Tested;
	 });
	 notify(TRN::Engine::Processor::Status::Priming);
	 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }

 void TRN::Engine::Processor::primed()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		// PrintThread{} << "waiting for rank " << handle->rank << " status " << status << " == " << TRN::Engine::Processor::Status::Priming << std::endl;
		 return status == TRN::Engine::Processor::Status::Priming;
	 });
	 notify(TRN::Engine::Processor::Status::Primed);
	 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }

 void TRN::Engine::Processor::tested()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 //PrintThread{} << "waiting for rank " << handle->rank << " status " << status << " == " << TRN::Engine::Processor::Status::Primed << std::endl;
		 return status == TRN::Engine::Processor::Status::Primed;
	 });
	 notify(TRN::Engine::Processor::Status::Tested);
	 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
 }

/* void TRN::Engine::Processor::ready()
 {
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		 PrintThread{} << "waiting for rank " << handle->rank << " status " << status << " == " << TRN::Engine::Processor::Status::Configured << std::endl;
		 return status == TRN::Engine::Processor::Status::Configured;
		 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
	 });
	 notify(TRN::Engine::Processor::Status::Ready);
 }*/
 void TRN::Engine::Processor::declare()
 {
	 //PrintThread{} << handle->rank << " entering " << __FUNCTION__ << std::endl;
	 wait([=](const TRN::Engine::Processor::Status &status)
	 {
		// PrintThread{} << "waiting for rank " << handle->rank << " status " << status << " == " << TRN::Engine::Processor::Status::Configured << std::endl;
		 return status == TRN::Engine::Processor::Status::Configured;
		 //PrintThread{} << handle->rank << " exiting " << __FUNCTION__ << std::endl;
	 });
 }


std::shared_ptr<TRN::Engine::Processor> TRN::Engine::Processor::create(const int &rank)
{
	return std::make_shared<TRN::Engine::Processor>(rank);
}

bool TRN::Engine::operator < (const std::shared_ptr<TRN::Engine::Processor> &left, const std::shared_ptr<TRN::Engine::Processor> &right)
{
	return left->get_latency() > right->get_latency();
}