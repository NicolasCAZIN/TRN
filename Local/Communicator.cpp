#include "stdafx.h"
#include "Communicator_impl.h"

TRN::Local::Communicator::Communicator(const int &max_rank) :
	TRN::Engine::Communicator(),
	handle(std::make_unique<Handle>())
{
	for (size_t k = 0; k < max_rank; k++)
		handle->queues.push_back(std::make_shared<TRN::Helper::Queue<std::shared_ptr<TRN::Local::Communicator::Handle::Blob>>>());
}

TRN::Local::Communicator::~Communicator()
{

	handle->workers.clear();
	handle.reset();
}
void TRN::Local::Communicator::dispose()
{
	for (auto queue : handle->queues)
		queue->invalidate();
	for (auto worker : handle->workers)
		worker->dispose();
}

int TRN::Local::Communicator::rank()
{
	return 0;
}
std::size_t TRN::Local::Communicator::size()
{
	return handle->queues.size();
}

void TRN::Local::Communicator::send(const int &destination, const TRN::Engine::Tag &tag, const std::string &data)
{
	std::lock_guard<std::recursive_mutex> lock(handle->mutex);
	if (destination >= handle->queues.size())
		throw std::runtime_error("Rank " + std::to_string(destination) + " overflows maximum declared rank " + std::to_string(handle->queues.size()));
	
	handle->queues[destination]->enqueue(std::make_shared<TRN::Local::Communicator::Handle::Blob>(tag, data));
}

std::string TRN::Local::Communicator::receive(const int &destination, const TRN::Engine::Tag &tag)
{
	//std::lock_guard<std::recursive_mutex> lock(handle->mutex);
	std::shared_ptr<TRN::Local::Communicator::Handle::Blob> blob;
	if (!handle->queues[destination]->dequeue(blob))
		throw std::runtime_error("received blob is invalid");
	if (blob->first != tag)
	{
		std::invalid_argument("tag mismatch");
	}
	return blob->second;
}

TRN::Engine::Tag TRN::Local::Communicator::probe(const int &destination)
{
	//std::lock_guard<std::recursive_mutex> lock(handle->mutex);
	//status(__FUNCTION__, "begin");
	std::shared_ptr<TRN::Local::Communicator::Handle::Blob> blob;

	if (!handle->queues[destination]->front(blob))
		throw std::runtime_error("received blob is invalid");

	return  blob->first;
}

void TRN::Local::Communicator::append(std::shared_ptr<TRN::Engine::Worker> &worker)
{
	handle->workers.push_back(worker);
}

std::shared_ptr<TRN::Local::Communicator> TRN::Local::Communicator::create(const int &max_rank)
{
	return std::make_shared<TRN::Local::Communicator>(max_rank);
}