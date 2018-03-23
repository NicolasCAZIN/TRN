#pragma once

#include "core_global.h"

#include "Helper/Visitor.h"
#include "Helper/Observer.h"
#include "Message.h"
#include "Initializer.h"
#include "Scheduling.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Reservoir :
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TESTED>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::PRIMED>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TRAINED>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::CONFIGURED>>,
			public TRN::Helper::Visitor<TRN::Core::Message::Payload<TRN::Core::Message::WEIGHTS>>,
			public TRN::Helper::Visitor<TRN::Core::Message::Payload<TRN::Core::Message::STATES>>,
			public TRN::Helper::Visitor<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>>,
			public TRN::Helper::Bridge<TRN::Backend::Driver>
		{

		protected :
			class Handle;
			std::unique_ptr<Handle> handle;

		protected :
			Reservoir(const std::shared_ptr<TRN::Backend::Driver> &driver,

				const std::size_t &stimulus, const std::size_t &prediction, const std::size_t &reservoir,
				const float &leak_rate,
				const float &initial_state_scale,
				const unsigned long &seed,
				const std::size_t &batch_size,
				const std::size_t &mini_batch_size
			);
			virtual ~Reservoir();

		public :
	
			void initialize(const std::shared_ptr<TRN::Core::Initializer> &feedforward, 
							const std::shared_ptr<TRN::Core::Initializer> &recurrent,
							const std::shared_ptr<TRN::Core::Initializer> &readout);
			void reset_readout();
			void initialize();
			void synchronize();
			void start();
			void stop();

		public :
			std::size_t get_batch_size();
			std::size_t get_reservoir_size();
			std::size_t get_stimulus_size();
			std::size_t get_prediction_size();
		public :
			void test(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Matrix> &incoming, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::size_t &preamble, const bool &autonomous_generation, const std::size_t &supplementary_generations);
			void train(const unsigned long long &evaluation_id, const std::shared_ptr<TRN::Core::Matrix> &incoming, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Scheduling> &scheduling);
				
		protected:
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS> &incoming) override;
			virtual void train(const std::shared_ptr<TRN::Core::Matrix> &incoming, const std::shared_ptr<TRN::Core::Matrix> &expected, const std::shared_ptr<TRN::Core::Scheduling> &scheduling,
							   std::shared_ptr<TRN::Core::Matrix> &states) = 0;



		public :
			virtual void visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::WEIGHTS>> &payload) override;
			virtual void visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::STATES>> &payload) override;
			virtual void visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload) override;
		};
	};
};