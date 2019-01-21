#pragma once

#include "encoder_global.h"
#include "Core/Encoder.h"

namespace TRN
{
	namespace Encoder
	{
		class ENCODER_EXPORT Model : public TRN::Core::Encoder
		{
		protected:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Model(const std::shared_ptr<TRN::Backend::Driver> &driver,  const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::shared_ptr<TRN::Core::Matrix> &cx,
				const std::shared_ptr<TRN::Core::Matrix> &cy,
				const std::shared_ptr<TRN::Core::Matrix> &width);
			virtual ~Model();

		public:
			virtual void encode(
				const std::shared_ptr<TRN::Core::Batch> &decoded_position,
				const unsigned long long &evaluation_id,
				std::shared_ptr<TRN::Core::Batch> &encoded_activations) override;

		public :
			static std::shared_ptr<Model> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::shared_ptr<TRN::Core::Matrix> &cx,
				const std::shared_ptr<TRN::Core::Matrix> &cy,
				const std::shared_ptr<TRN::Core::Matrix> &width);
		};
	};
};

