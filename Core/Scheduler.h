#pragma once

#include "Message.h"
#include "Container.h"
#include "Scheduling.h"
#include "Backend/Driver.h"
#include "Helper/Observer.h"
#include "Helper/Visitor.h"
#include "Helper/Delegate.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Scheduler :
			public TRN::Helper::Bridge<TRN::Backend::Driver>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>>,
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::SET>>,
			public TRN::Helper::Delegator<TRN::Core::Container>
		{
		protected :
			Scheduler(const std::shared_ptr<TRN::Backend::Driver> &driver);
		public :
			virtual ~Scheduler();
		};
	};
};
