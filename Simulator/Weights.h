#pragma once

#include "simulator_global.h"
#include "Core/Simulator.h"
#include "Helper/Decorator.h"

namespace TRN
{
	namespace Simulator
	{
		class SIMULATOR_EXPORT Weights :
			public TRN::Helper::Decorator<TRN::Core::Simulator>,
			public TRN::Helper::Observer<TRN::Core::Message::Payload<TRN::Core::Message::CYCLES>>

		{
		private:
			class Handle;
			mutable std::unique_ptr<Handle> handle;

		public:
			Weights(const std::shared_ptr<TRN::Core::Simulator> &decorated,
				const std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initialization, const bool &train);
			~Weights();
		public:
			virtual const std::shared_ptr<TRN::Core::Matrix> retrieve_sequence(const std::string &label, const std::string &tag) override;
			virtual const std::shared_ptr<TRN::Core::Set> retrieve_set(const std::string &label, const std::string &tag) override;
		public:
			virtual const std::shared_ptr<TRN::Core::Reservoir> get_reservoir() override;
			virtual const std::shared_ptr<TRN::Core::Loop> get_loop() override;
			virtual const std::shared_ptr<TRN::Core::Scheduler> get_scheduler() override;
			virtual const std::vector<std::shared_ptr<TRN::Core::Mutator>> get_mutators() override;

		public:
			virtual void set_feedforward(const std::shared_ptr<TRN::Core::Initializer> &feedforward) override;
			virtual void set_recurrent(const std::shared_ptr<TRN::Core::Initializer> &recurrent) override;
			virtual void set_feedback(const std::shared_ptr<TRN::Core::Initializer> &feedback) override;
			virtual void set_readout(const std::shared_ptr<TRN::Core::Initializer> &readout) override;
			virtual void set_reservoir(const std::shared_ptr<TRN::Core::Reservoir> &reservoir) override;
			virtual void set_scheduler(const std::shared_ptr<TRN::Core::Scheduler> &scheduler) override;
			virtual void set_loop(const std::shared_ptr<TRN::Core::Loop> &loop) override;
			virtual void append_measurement(const std::shared_ptr<TRN::Core::Measurement::Abstraction> &measurement) override;
			virtual void append_mutator(const std::shared_ptr<TRN::Core::Mutator> &mutator) override;
		public:
			virtual void declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Matrix> &sequence) override;
			virtual void declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Set> &set) override;
			virtual void train(const std::string &sequence, const std::string &incoming, const std::string &expected) override;
			virtual void test(const std::string &sequence, const std::string &incoming, const std::string &expected, const std::size_t &preamble, const std::size_t &supplementary_generations) override;

			virtual void initialize() override;
			virtual void uninitialize() override;

		public :
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::CYCLES> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::TESTED> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::PRIMED> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::TRAINED> &payload) override;
		public:
			static std::shared_ptr<Weights> create(const std::shared_ptr<TRN::Core::Simulator> decorated,
				const std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &data, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initialization, const bool &train);
		};
	};
};

