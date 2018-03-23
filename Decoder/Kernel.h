#pragma once

#include "decoder_global.h"
#include "Core/Decoder.h"

namespace TRN
{
	namespace Decoder
	{
		class DECODER_EXPORT Kernel : public TRN::Core::Decoder
		{
		protected:
			class Handle;
			std::unique_ptr<Handle> handle;

		protected:
			Kernel(const std::shared_ptr<TRN::Backend::Driver> &driver,
				const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::size_t &rows, const std::size_t &cols,
				const std::pair<float, float> &arena_x, const std::pair<float, float> &arena_y,
				const float &sigma, const float &radius, const float &angle, const float &scale, const unsigned long &seed);
		public :
			virtual ~Kernel();
		public :
			virtual void decode(
				const std::shared_ptr<TRN::Core::Batch> &previous_position,
				const std::shared_ptr<TRN::Core::Batch> &current_position,
				const std::shared_ptr<TRN::Core::Batch> &predicted_activations,
				std::shared_ptr<TRN::Core::Batch> &decoded_position) override;

		protected :
			virtual void location_probability(
				const std::shared_ptr<TRN::Core::Batch> &previous_position,
				const std::shared_ptr<TRN::Core::Batch> &current_position,
				const std::shared_ptr<TRN::Core::Batch> &predicted_activations, 
				std::shared_ptr<TRN::Core::Batch> &location_probability) = 0;
		};
	};
};
