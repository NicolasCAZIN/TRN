#pragma once

#include "core_global.h"

#include "Reservoir.h"
#include "Scheduler.h"
#include "Loop.h"
#include "Measurement.h"
#include "Container.h"
#include "Mutator.h"
#include "Helper/Delegate.h"

namespace TRN
{
	namespace Core
	{
		class CORE_EXPORT Simulator :
			public std::enable_shared_from_this<TRN::Core::Simulator>,
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::CYCLES>>,
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::TESTED>>,
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::PRIMED>>,
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::TRAINED>>,
			public TRN::Helper::Delegate<TRN::Core::Container>
		{
		protected :
			Simulator();
		public :
			virtual ~Simulator();
		public :
			virtual const std::shared_ptr<TRN::Core::Reservoir> get_reservoir() = 0;
			virtual const std::shared_ptr<TRN::Core::Loop> get_loop() = 0;
			virtual const std::shared_ptr<TRN::Core::Scheduler> get_scheduler() = 0;
		public:
			virtual void set_feedforward(const std::shared_ptr<TRN::Core::Initializer> &feedforward) = 0;
			virtual void set_recurrent(const std::shared_ptr<TRN::Core::Initializer> &recurrent) = 0;
			virtual void set_feedback(const std::shared_ptr<TRN::Core::Initializer> &feedback) = 0;
			virtual void set_readout(const std::shared_ptr<TRN::Core::Initializer> &readout) = 0;
			virtual void set_reservoir(const std::shared_ptr<TRN::Core::Reservoir> &reservoir) = 0;
			virtual void set_scheduler(const std::shared_ptr<TRN::Core::Scheduler> &scheduler) = 0;
			virtual void set_loop(const std::shared_ptr<TRN::Core::Loop> &loop) = 0;
			virtual void append_measurement(const std::shared_ptr<TRN::Core::Measurement::Abstraction> &measurement) = 0;
			virtual void append_mutator(const std::shared_ptr<TRN::Core::Mutator> &mutator) = 0;
		public :
			virtual void declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Matrix> &sequence) = 0;
			virtual void declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Set> &batch) = 0;
			virtual void train(const std::string &sequences, const std::string &incoming, const std::string &expected) = 0;
			virtual void test(const std::string &sequence, const std::string &incoming, const std::string &expected, const std::size_t &preamble, const std::size_t &supplementary_generations) = 0;
			virtual void initialize() = 0;
			virtual void uninitialize() = 0;
		};
	};
};
