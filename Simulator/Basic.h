#pragma once

#include "simulator_global.h"
#include "Core/Simulator.h"

namespace TRN
{
	namespace Simulator
	{
		class SIMULATOR_EXPORT Basic : 
			public TRN::Core::Simulator,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::SET>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TEST>>,
			public TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::TARGET_SEQUENCE>>
		{

		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Basic( const std::function<void()> &trained, const std::function<void()> &primed, const std::function<void()> &tested);
			~Basic();

		public :
			virtual const std::shared_ptr<TRN::Core::Set> retrieve_set(const std::string &label, const std::string &tag) override;
			virtual const std::shared_ptr<TRN::Core::Matrix> retrieve_sequence(const std::string &label, const std::string &tag) override;
		public :

			virtual const std::shared_ptr<TRN::Core::Reservoir> get_reservoir() override;
			virtual const std::shared_ptr<TRN::Core::Loop> get_loop() override;
			virtual const std::shared_ptr<TRN::Core::Decoder> get_decoder() override;
			virtual const std::shared_ptr<TRN::Core::Encoder> get_encoder() override;
			virtual const std::shared_ptr<TRN::Core::Scheduler> get_scheduler() override;
			virtual const std::vector<std::shared_ptr<TRN::Core::Mutator>> get_mutators() override;

		public:
			virtual void set_feedforward(const std::shared_ptr<TRN::Core::Initializer> &feedforward) override;
			virtual void set_recurrent(const std::shared_ptr<TRN::Core::Initializer> &recurrent) override;
			virtual void set_readout(const std::shared_ptr<TRN::Core::Initializer> &readout) override;
			virtual void set_reservoir(const std::shared_ptr<TRN::Core::Reservoir> &reservoir) override;
			virtual void set_scheduler(const std::shared_ptr<TRN::Core::Scheduler> &scheduler) override;
			virtual void set_loop(const std::shared_ptr<TRN::Core::Loop> &loop) override;
			virtual void set_decoder(const std::shared_ptr<TRN::Core::Decoder> &decoder) override;
			virtual void set_encoder(const std::shared_ptr<TRN::Core::Encoder> &encoder) override;
			virtual void append_measurement(const std::shared_ptr<TRN::Core::Measurement::Abstraction> &measurement) override;
			virtual void append_mutator(const std::shared_ptr<TRN::Core::Mutator> &mutator) override;

		public:
			virtual void declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Matrix> &sequence) override;
			virtual void declare(const std::string &label, const std::string &tag, const std::shared_ptr<TRN::Core::Set> &batch) override;
			virtual void train(const unsigned long long &evaluation_id, const std::string &label, const std::string &incoming, const std::string &expected, const bool &reset_readout) override;
			virtual void test(const unsigned long long &evaluation_id, const std::string &sequence, const std::string &incoming, const std::string &expected, const std::size_t &preamble, const bool &autonomous, const std::size_t &supplementary_generations) override;
			virtual void initialize() override;
			virtual void uninitialize() override;

		public :
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::SCHEDULING> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::TESTED> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::CONFIGURED> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::PRIMED> &payload) override;
			virtual void update(const TRN::Core::Message::Payload<TRN::Core::Message::TRAINED> &payload) override;
		public:
			static std::shared_ptr<Basic> create( const std::function<void()> &trained, const std::function<void()> &primed, const std::function<void()> &tested);

		private :
			static const std::string key(const std::string &label, const std::string &tag);

		};
	};
};
