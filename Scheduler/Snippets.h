#pragma once

#include "scheduler_global.h"
#include "Core/Scheduler.h"

namespace TRN
{
	namespace Scheduler
	{
		class SCHEDULER_EXPORT Snippets : public TRN::Core::Scheduler
		{
		private :
			class Handle;
			mutable std::unique_ptr<Handle> handle;

		public:
			Snippets(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag = "");
			virtual ~Snippets();

		public :
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::SET> &payload);

		public :
			static std::shared_ptr<Snippets> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag = "");
		};

	};
};
