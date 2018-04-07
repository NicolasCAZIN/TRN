#pragma once

#include "core_global.h"
#include "Batch.h"
#include "Helper/Visitor.h"
#include "Helper/Bridge.h"
#include "Helper/Observer.h"
#include "Core/Message.h"
#include "Message.h"
#include "Backend/Driver.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Encoder :
			public TRN::Helper::Bridge<TRN::Backend::Driver>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::POSITION>>
		{

		protected:
			class Handle;
			std::unique_ptr<Handle> handle;

		protected:
			Encoder(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size);

		public:
			virtual ~Encoder();

		public:
			virtual void encode(
				const std::shared_ptr<TRN::Core::Batch> &decoded_position,
				const unsigned long long &evaluation_id,
				std::shared_ptr<TRN::Core::Batch> &encoded_activations) = 0;
		};
	};
};