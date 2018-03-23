#pragma once

#include "reservoir_global.h"
#include "Core/Reservoir.h"

namespace TRN
{
	namespace Reservoir
	{
		class RESERVOIR_EXPORT WidrowHoff : 
			public TRN::Core::Reservoir
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;
		public:
			WidrowHoff(const std::shared_ptr<TRN::Backend::Driver> &driver,
				const std::size_t &stimulus, const std::size_t &prediction, const std::size_t &reservoir,
				const float &leak_rate,
				const float &initial_state_scale,
				const float &learning_rate,
				const unsigned long &seed,
				const std::size_t &batch_size, const std::size_t &mini_batch_size);

		public :
			virtual void train(const std::shared_ptr<TRN::Core::Matrix> &incoming, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Scheduling> &scheduling,
								std::shared_ptr<TRN::Core::Matrix> &states) override;
			virtual void visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload) override;
		public :
			static std::shared_ptr<WidrowHoff> create(const std::shared_ptr<TRN::Backend::Driver> &driver, 
				const std::size_t &stimulus, const std::size_t &prediction, const std::size_t &reservoir,
				const float &leak_rate,
				const float &initial_state_scale,
				const float &learning_rate,
				const unsigned long &seed,
				const std::size_t &batch_size, const std::size_t &mini_batch_size);
			
		};

	};
};
