#include "stdafx.h"
#include "Helper/Logger.h"
#include "Manager_impl.h"

TRN::Engine::Manager::Manager(const std::size_t &size):
	handle(std::make_unique<Handle>())
{
	TRACE_LOGGER;
	if (size <= 1)
		throw std::runtime_error("At least, one processor is required");
	for (std::size_t k = 1; k < size; k++)
	{
		auto processor = TRN::Engine::Processor::create(k);

		handle->processors.push_back(processor);
		handle->updated[k] = false;
	}
}

void TRN::Engine::Manager::start()
{
	TRACE_LOGGER;
	for (auto processor : handle->processors)
	{
		processor->start();
	}
	handle->running = true;
	handle->deallocator = std::thread([&]()
	{
		unsigned long long simulation_id;
		while (handle->to_deallocate.dequeue(simulation_id))
		{
			std::unique_lock<std::mutex> lock(handle->mutex);

			//INFORMATION_LOGGER <<   "deallocate " << id ;
			handle->available.emplace(handle->associated[simulation_id]);
			handle->associated.erase(simulation_id);

			lock.unlock();
			handle->condition.notify_all();
		}
		std::unique_lock<std::mutex> lock(handle->mutex);
		handle->running = false;

		lock.unlock();
		handle->condition.notify_one();
	});
	handle->deallocator.detach();
}

void TRN::Engine::Manager::stop()
{
	TRACE_LOGGER;

	for (auto processor : handle->processors)
	{
		processor->stop();

	}
	handle->to_deallocate.invalidate();

	std::unique_lock<std::mutex> lock(handle->mutex);
	while (handle->running)
		handle->condition.wait(lock);
}

TRN::Engine::Manager::~Manager()
{
	TRACE_LOGGER;

	handle.reset();
}
std::vector<std::shared_ptr<TRN::Engine::Processor>> TRN::Engine::Manager::get_processors()
{
	TRACE_LOGGER;
	return handle->processors;
}

void TRN::Engine::Manager::update_processor(const int &rank, const std::string host, const unsigned int &index, const std::string name)
{
	TRACE_LOGGER;
	std::unique_lock<std::mutex> lock(handle->mutex);
	auto processor = handle->processors[rank - 1];
	if (!handle->updated[rank - 1])
	{
		processor->set_host(host);
		processor->set_name(name);
		processor->set_index(index);

		handle->updated[rank - 1] = true;
		handle->available.emplace(processor);
		lock.unlock();

	}
	handle->condition.notify_all();
}

/*void TRN::Engine::Manager::wait_not_allocated()
{
	std::unique_lock<std::mutex> lock(handle->mutex);

	while (handle->available.size() < handle->processors.size())
	{
		//INFORMATION_LOGGER <<   handle->associated.size() << " simulations still pending" ;
		handle->condition.wait(lock);
	}
	//INFORMATION_LOGGER <<   "no more simulations pending" ;
}*/
std::shared_ptr<TRN::Engine::Processor> TRN::Engine::Manager::allocate(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	std::unique_lock<std::mutex> lock(handle->mutex);
	if (handle->associated.find(simulation_id) != handle->associated.end())
	{
		throw std::invalid_argument("Simulator #" + std::to_string(simulation_id) + " was already associated");
	}

	while (handle->available.empty())
	{
		//INFORMATION_LOGGER <<   "waiting for non empty" ;
		handle->condition.wait(lock);
	}

	auto processor = handle->available.top();
	handle->available.pop();
	handle->associated[simulation_id] = processor;

	lock.unlock();
	handle->condition.notify_all();
	
	return processor;
}

void TRN::Engine::Manager::synchronize()
{
	TRACE_LOGGER;
	std::unique_lock<std::mutex> lock(handle->mutex);

	while (!handle->associated.empty())
		handle->condition.wait(lock);

	lock.unlock();
}
void TRN::Engine::Manager::deallocate(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	handle->to_deallocate.enqueue(simulation_id);
}

std::shared_ptr<TRN::Engine::Processor> TRN::Engine::Manager::retrieve(const unsigned long long &simulation_id)
{
	TRACE_LOGGER;
	std::unique_lock<std::mutex> lock(handle->mutex);
	
	while (handle->associated.find(simulation_id) == handle->associated.end())
	{
		//INFORMATION_LOGGER <<   "waiting for " << id <<  " non empty" ;
		handle->condition.wait(lock);
	}

	return (handle->associated[simulation_id]);
}

std::shared_ptr<TRN::Engine::Manager> TRN::Engine::Manager::create(const std::size_t &size)
{
	TRACE_LOGGER;
	return std::make_shared<TRN::Engine::Manager>(size);
}