#pragma once

#include "encoder_global.h"
#include "Core/Encoder.h"

namespace TRN
{
	namespace Encoder
	{
		class ENCODER_EXPORT Custom : public TRN::Core::Encoder
		{
		protected:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Custom(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus);
			virtual ~Custom();

		public:
			virtual void encode(
				const std::shared_ptr<TRN::Core::Batch> &decoded_position,
				const unsigned long long &evaluation_id,
				std::shared_ptr<TRN::Core::Batch> &encoded_activations) override;

		public:
			static std::shared_ptr<Custom> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
				std::function<void(const unsigned long long &evaluation_id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus);
		};
	};
};

