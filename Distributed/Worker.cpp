#include "stdafx.h"
#include "Worker_impl.h"


TRN::Distributed::Worker::Worker(const std::string &name, const int &rank) :
	handle(std::make_unique<TRN::Distributed::Worker::Handle>(name, rank))
{
	std::cout << "rank is " << handle->rank << std::endl;
}

TRN::Distributed::Worker::~Worker()
{
	handle.reset();
}
void TRN::Distributed::Worker::start()
{
	stop();
	handle->sender = std::thread([&]
	{
		std::cout << "loader " << handle->rank << " started" << std::endl;
		bool stop = false;
		while (!stop)
		{
			try
			{
				std::unique_lock<std::mutex> lock(handle->send);
				
				while (!handle->pending.empty())
					handle->jobs_available.wait(lock);

				while (handle->jobs.empty())
					handle->jobs_available.wait(lock);
		

				auto job = handle->jobs.front();
				handle->jobs.pop();
				std::cout << "processing job for rank " << handle->rank << std::endl;
				if (handle->jobs.empty() && handle->stop_requested)
				{
					stop = true;
				}
				lock.unlock();
				job();
			}
			catch (std::exception &e)
			{
				std::cerr << e.what() << std::endl;
				
			}
		}
		std::cout << "loader " << get_rank() << " stopped" << std::endl;
	});

}


void TRN::Distributed::Worker::post(const std::function<void()> &job)
{
	std::unique_lock<std::mutex> lock(handle->send);
	handle->jobs.emplace(job);
	handle->jobs_available.notify_one();
}

void TRN::Distributed::Worker::remove_pending(const size_t &number)
{
	std::unique_lock<std::mutex> lock(handle->send);
	if (handle->pending.find(number) == handle->pending.end())
	{
		throw std::invalid_argument("job #" + std::to_string(number) + " is not a pending job");
	}
	handle->pending.erase(number);


		handle->jobs_available.notify_one();
}

void TRN::Distributed::Worker::add_pending(const size_t &number)
{
	std::unique_lock<std::mutex> lock(handle->send);

	if (handle->pending.find(number) != handle->pending.end())
	{
		throw std::invalid_argument("job #" + std::to_string(number) + " is already a pending job");
	}

	handle->pending.insert(number);
	handle->jobs_available.notify_one();
}


void TRN::Distributed::Worker::stop()
{
	post([&]()
	{
		std::unique_lock<std::mutex> lock(handle->send);
		while (!handle->pending.empty())
			handle->jobs_available.wait(lock);

		handle->stop_requested = true;
	});
	if (handle->sender.joinable())
		handle->sender.join();
	handle->stop_requested = false;

}
std::string TRN::Distributed::Worker::get_name()
{
	std::unique_lock<std::mutex> lock(handle->send);
	return handle->name;
}

int TRN::Distributed::Worker::get_rank()
{
	std::unique_lock<std::mutex> lock(handle->send);
	std::cout << "RANK " << handle->rank << std::endl;
	return handle->rank;
}

float TRN::Distributed::Worker::get_latency()
{
	std::unique_lock<std::mutex> lock(handle->send);
	return handle->latency;
}

void TRN::Distributed::Worker::set_latency(const float &latency)
{
	std::unique_lock<std::mutex> lock(handle->send);
	handle->latency = latency;
}

bool TRN::Distributed::operator < (const std::shared_ptr<TRN::Distributed::Worker> &left, const std::shared_ptr<TRN::Distributed::Worker> &right)
{
	return left->get_latency() > right->get_latency();
}