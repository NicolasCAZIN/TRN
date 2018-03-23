#pragma once

#include "core_global.h"
#include "Helper/Observer.h"
#include "Helper/Delegate.h"
#include "Helper/Visitor.h"
#include "Message.h"
#include "Container.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Loop : 
			public TRN::Helper::Bridge<TRN::Backend::Driver>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::POSITION>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TARGET_TRAJECTORY>>,
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION>>,
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::TEST>>,
			public TRN::Helper::Delegator<TRN::Core::Container>,
			public TRN::Helper::Visitor<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>>
		{
		protected :
			class Handle;
			std::unique_ptr<Handle> handle;

		protected :
			Loop(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size);

		public :
			virtual ~Loop();
		public :
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::TEST> &payload) override;

		};
	};
};