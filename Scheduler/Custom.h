#pragma once

#include "scheduler_global.h"
#include "Core/Scheduler.h"

namespace TRN
{
	namespace Scheduler
	{
		class SCHEDULER_EXPORT Custom : public TRN::Core::Scheduler
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Custom(const std::shared_ptr<TRN::Backend::Driver> &driver, 
				const std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)> &request,
				std::function<void(const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)> &reply,
				const std::string &tag);
			~Custom();

		protected :
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload) override;
			

		public:
			static std::shared_ptr<Custom> create(const std::shared_ptr<TRN::Backend::Driver> &driver, 
				const std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)> &request,
				std::function<void(const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)> &reply, const std::string &tag);
		};

	};
};