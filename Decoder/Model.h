#pragma once

#include "decoder_global.h"
#include "Kernel.h"

namespace TRN
{
	namespace Decoder
	{
		class DECODER_EXPORT Model : public Kernel
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Model(
				const std::shared_ptr<TRN::Backend::Driver> &driver, 
				const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::size_t &rows, const std::size_t &cols,
				const std::pair<float, float> &arena_x, const std::pair<float, float> &arena_y,
				const float &sigma, const float &radius, const float &angle, const float &scale, const unsigned long &seed,
				const std::shared_ptr<TRN::Core::Matrix> &cx,
				const std::shared_ptr<TRN::Core::Matrix> &cy,
				const std::shared_ptr<TRN::Core::Matrix> &width
		);
		public :
			virtual ~Model();

		protected:
			virtual void location_probability(
				const std::shared_ptr<TRN::Core::Batch> &previous_position,
				const std::shared_ptr<TRN::Core::Batch> &current_position,
				const std::shared_ptr<TRN::Core::Batch> &predicted_activations,
				std::shared_ptr<TRN::Core::Batch> &location_probability) override;

		public:

			virtual void visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload) override;

		public:
			static std::shared_ptr<TRN::Decoder::Model> create(const std::shared_ptr<TRN::Backend::Driver> &driver,
				const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::size_t &rows, const std::size_t &cols,
				const std::pair<float, float> &arena_x, const std::pair<float, float> &arena_y,
				const float &sigma, const float &radius, const float &angle, const float &scale, const unsigned long &seed,
				const std::shared_ptr<TRN::Core::Matrix> &cx,
				const std::shared_ptr<TRN::Core::Matrix> &cy,
				const std::shared_ptr<TRN::Core::Matrix> &width);
		};
	};
};
