#pragma once

#include "decoder_global.h"
#include "Core/Decoder.h"

namespace TRN
{
	namespace Decoder
	{
		class DECODER_EXPORT Linear : public TRN::Core::Decoder
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Linear(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, const std::shared_ptr<TRN::Core::Matrix> &cx, const std::shared_ptr<TRN::Core::Matrix> &cy);
		public :
			virtual ~Linear();

		public :
			virtual void decode(
				const std::shared_ptr<TRN::Core::Batch> &previous_position,
				const std::shared_ptr<TRN::Core::Batch> &current_position,
				const std::shared_ptr<TRN::Core::Batch> &predicted_activations,
				std::shared_ptr<TRN::Core::Batch> &decoded_position) override;
			virtual void visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload) override;

		public :
			static std::shared_ptr<TRN::Decoder::Linear> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, const std::shared_ptr<TRN::Core::Matrix> &cx, const std::shared_ptr<TRN::Core::Matrix> &cy);
		};
	};
};
