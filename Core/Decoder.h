#pragma once

#include "core_global.h"
#include "Batch.h"
#include "Helper/Visitor.h"
#include "Helper/Bridge.h"
#include "Message.h"
#include "Backend/Driver.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Decoder :
			public TRN::Helper::Bridge<TRN::Backend::Driver>,
			public TRN::Helper::Visitor<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>>
		{

		protected :
			class Handle;
			std::unique_ptr<Handle> handle;

		protected :
			Decoder(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size);
		public :
			virtual ~Decoder();

		public:
			virtual void decode(
				const std::shared_ptr<TRN::Core::Batch> &previous_position,
				const std::shared_ptr<TRN::Core::Batch> &current_position,
				const std::shared_ptr<TRN::Core::Batch> &predicted_activations, 
				std::shared_ptr<TRN::Core::Batch> &decoded_position) = 0;

		};
	};
};