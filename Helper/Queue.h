#pragma once

#include "helper_global.h"

namespace TRN
{
	namespace Helper
	{
		template<typename Data>
		class  Queue
		{
		private :
			std::queue<Data> queue;
			std::mutex mutex;
			std::condition_variable cond;
			bool valid;
		public :
			Queue() :
				valid(true)
			{

			}

		public :
			bool empty()
			{
				std::unique_lock<std::mutex> lock(mutex);
				auto is_empty = !queue.empty();
				lock.unlock();
				return is_empty;
			}

			void invalidate()
			{
				std::unique_lock<std::mutex> lock(mutex);
				valid = false;
				lock.unlock();
				cond.notify_one();
			}

			void enqueue(const Data &data)
			{
				std::unique_lock<std::mutex> lock(mutex);
				queue.push(data);
				lock.unlock();
				cond.notify_one();
			}

			bool front(Data &data)
			{
				std::unique_lock<std::mutex> lock(mutex);
				while (true)
				{
					while (valid && queue.empty())
					{
						cond.wait(lock);
					}
					if (!queue.empty())
					{
						data = queue.front();
						return true;
					}
					else if (!valid)
					{
						return false;
					}
				}

			}

			bool dequeue(Data &data)
			{
				std::unique_lock<std::mutex> lock(mutex);
				while (true)
				{
					while (valid && queue.empty())
					{
						cond.wait(lock);
					}
					if (!queue.empty())
					{
						data = queue.front();
						queue.pop();
						return true;
					}
					else if (!valid)
					{
						return false;
					}
				}
			
			}
		};
	};
};
