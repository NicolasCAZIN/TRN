#include "stdafx.h"
#include "Manager_impl.h"

TRN::Engine::Manager::Manager(const std::size_t &size):
	handle(std::make_unique<Handle>())
{
	if (size <= 1)
		throw std::runtime_error("At least, one processor is required");
	for (std::size_t k = 1; k < size; k++)
		handle->processors.push_back(TRN::Engine::Processor::create(k));

	for (auto processor : handle->processors)
	{
		handle->available.emplace(processor);
	}
	handle->deallocator = std::thread([&]()
	{
		unsigned int id;
		while (handle->to_deallocate.dequeue(id))
		{
			std::unique_lock<std::mutex> lock(handle->mutex);

			//PrintThread{} << "deallocate " << id << std::endl;
			handle->available.emplace(handle->associated[id]);
			handle->associated.erase(id);

			lock.unlock();
			handle->condition.notify_one();
		}
	});
}

TRN::Engine::Manager::~Manager()
{

	handle->to_deallocate.invalidate();


	if (handle->deallocator.joinable())
		handle->deallocator.join();
	handle.reset();
}
std::vector<std::shared_ptr<TRN::Engine::Processor>> TRN::Engine::Manager::get_processors()
{
	return handle->processors;
}

void TRN::Engine::Manager::wait_not_allocated()
{
	std::unique_lock<std::mutex> lock(handle->mutex);

	while (handle->available.size() < handle->processors.size())
	{
		//PrintThread{} << handle->associated.size() << " simulations still pending" << std::endl;
		handle->condition.wait(lock);
	}
	//PrintThread{} << "no more simulations pending" << std::endl;
}
std::shared_ptr<TRN::Engine::Processor> TRN::Engine::Manager::allocate(const unsigned int &id)
{
	std::unique_lock<std::mutex> lock(handle->mutex);
	if (handle->associated.find(id) != handle->associated.end())
	{
		throw std::invalid_argument("Simulator #" + std::to_string(id) + " was already associated");
	}

	while (handle->available.empty())
	{
		//PrintThread{} << "waiting for non empty" << std::endl;
		handle->condition.wait(lock);
	}

	auto processor = handle->available.top();
	handle->available.pop();
	handle->associated[id] = processor;

	lock.unlock();
	handle->condition.notify_one();
	
	return processor;
}

void TRN::Engine::Manager::deallocate(const unsigned int &id)
{
	handle->to_deallocate.enqueue(id);
}

std::shared_ptr<TRN::Engine::Processor> TRN::Engine::Manager::retrieve(const unsigned int &id)
{
	std::unique_lock<std::mutex> lock(handle->mutex);
	
	while (handle->associated.find(id) == handle->associated.end())
	{
		//PrintThread{} << "waiting for " << id <<  " non empty" << std::endl;
		handle->condition.wait(lock);
	}

	return (handle->associated[id]);
}

std::shared_ptr<TRN::Engine::Manager> TRN::Engine::Manager::create(const std::size_t &size)
{
	return std::make_shared<TRN::Engine::Manager>(size);
}