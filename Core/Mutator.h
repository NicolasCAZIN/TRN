#pragma once

#include "core_global.h"
#include "Helper/Observer.h"
#include "Message.h"

namespace TRN
{
	namespace Core
	{
		class Scheduling;
		class CORE_EXPORT Mutator : 
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>>
		{

		};

	};
};
