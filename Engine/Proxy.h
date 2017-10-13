#pragma once

#include "Node.h"
#include "Broker.h"
#include "Helper/Visitor.h"
namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Proxy : 
			public std::enable_shared_from_this<TRN::Engine::Proxy>,
			public TRN::Engine::Broker, 
			public TRN::Engine::Node
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Proxy(const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy, const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor);
			virtual ~Proxy();

		protected :
			virtual void uninitialize() override;
			virtual void initialize() override;
		public :
			virtual void start() override;

		protected:
			virtual void callback_completed() override;
			virtual void callback_configured(const unsigned long long &id);
			virtual void callback_ack(const unsigned long long &id, const std::size_t &number, const bool &success, const std::string &cause) override;
			virtual void callback_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name) override;
			virtual void callback_allocated(const unsigned long long &id, const int &rank) override;
			virtual void callback_deallocated(const unsigned long long &id, const int &rank) override;
			virtual void callback_quit(const int &rank) override;
			virtual void callback_trained(const unsigned long long &id) override;
			virtual void callback_primed(const unsigned long long &id) override;
			virtual void callback_tested(const unsigned long long &id) override;
			virtual void callback_error(const std::string &message) override;
			virtual void callback_information(const std::string &message) override;
			virtual void callback_warning(const std::string &message) override;

			virtual void callback_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_readout_custom(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_measurement_position_custom(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) override;

			virtual void callback_performances(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second) override;
			virtual void callback_states(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override;
			virtual void callback_weights(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) override;
			virtual void callback_position(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols) override;
			virtual void callback_stimulus(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols) override;

			virtual void callback_mutator(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations) override;
			virtual void callback_scheduler(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations) override;
			virtual void callback_scheduling(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations) override;

			virtual void callback_feedforward(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_feedback(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_readout(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) override;
			virtual void callback_recurrent(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols) override;


		public:
			//virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::READY> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::COMPLETED> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::ALLOCATE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::DEALLOCATE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::TRAIN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::TEST> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SEQUENCE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::DECLARE_SET> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_STATES> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_WEIGHTS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_PERFORMANCES> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SETUP_SCHEDULING> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_BEGIN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_END> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_MEAN_SQUARE_ERROR> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_FRECHET_DISTANCE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_READOUT_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_MEAN_SQUARE_ERROR> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_FRECHET_DISTANCE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MEASUREMENT_POSITION_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RESERVOIR_WIDROW_HOFF> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_COPY> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_SPATIAL_FILTER> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_LOOP_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_TILED> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_SNIPPETS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_SCHEDULER_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_SHUFFLE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_REVERSE> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_PUNCH> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_MUTATOR_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_UNIFORM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_GAUSSIAN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDFORWARD_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_UNIFORM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_GAUSSIAN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_FEEDBACK_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_UNIFORM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_GAUSSIAN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_RECURRENT_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_UNIFORM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_GAUSSIAN> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::CONFIGURE_READOUT_CUSTOM> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::POSITION> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::STIMULUS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::SCHEDULING> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDFORWARD_WEIGHTS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::RECURRENT_WEIGHTS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::FEEDBACK_WEIGHTS> &message) override;
			virtual void process(const TRN::Engine::Message<TRN::Engine::Tag::READOUT_WEIGHTS> &message) override;

			static std::shared_ptr<Proxy> create(const std::shared_ptr<TRN::Engine::Communicator> &frontend_proxy, const std::shared_ptr<TRN::Engine::Communicator> &proxy_workers, const std::shared_ptr<TRN::Helper::Visitor<TRN::Engine::Proxy>> &visitor);
		};
	};
};

