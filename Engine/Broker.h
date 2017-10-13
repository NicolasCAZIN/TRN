#pragma once

#include "engine_global.h"
#include "Messages.h"
#include "Manager.h"
#include "Task.h"
#include "Communicator.h"
#include "Executor.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Broker : public TRN::Engine::Task
		{
		protected :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Broker(const std::shared_ptr<TRN::Engine::Communicator> &communicator);
		public :
			virtual ~Broker();

		public :
			void halt();

		protected :
			virtual void initialize() override;
			virtual void uninitialize() override;
			virtual void body() override;


		protected :
			void	completed();
	
		protected :
			//void ready(const unsigned long long &id);
		public  :
			void 	allocate(const unsigned long long &id);
			void 	deallocate(const unsigned long long &id);
			void 	train(const unsigned long long &id, const std::string &label, const std::string &incoming, const std::string &expected);
			void 	test(const unsigned long long &id, const std::string &sequence, const std::string &incoming, const std::string &expected, const unsigned int &preamble, const unsigned int &supplementary_generations);
			void 	declare_sequence(const unsigned long long &id, const std::string &label, const std::string &tag,
				const std::vector<float> &sequence, const std::size_t &observations);
			void 	declare_set(const unsigned long long &id, const std::string &label, const std::string &tag,
				const std::vector<std::string> &labels);
			void 	setup_states(const unsigned long long &id, const bool &train, const bool &prime, const bool &generate);
			void 	setup_weights(const unsigned long long &id, const bool &initilization, const bool &train);
			void 	setup_performances(const unsigned long long &id, const bool &train, const bool &prime, const bool &generate);
			void 	setup_scheduling(const unsigned long long &id);

			void	configure_begin(const unsigned long long &id);
			void	configure_end(const unsigned long long &id);

			void 	configure_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &batch_size);
			void  	configure_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &batch_size);
			void  	configure_measurement_readout_custom(const unsigned long long &id, const std::size_t &batch_size);

			void  	configure_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &batch_size);
			void  	configure_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &batch_size);
			void  	configure_measurement_position_custom(const unsigned long long &id, const std::size_t &batch_size);

			void 	configure_reservoir_widrow_hoff(const unsigned long long &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate,
				const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size);
			/*virtual void 	configure_reservoir_online_force(const unsigned long long &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate,
			const float &initial_state_scale, const float &learning_rate) = 0;*/
			/*	virtual void 	configure_reservoir_offline_svd(const unsigned long long &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate,
			const float &initial_state_scale, const float &learning_rate) = 0;*/
			void 	configure_loop_copy(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size);
			void 	configure_loop_spatial_filter(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size, const unsigned long &seed,
				const std::size_t &rows, const std::size_t &cols,
				const std::pair<float, float> &x, const std::pair<float, float> &y,
				const std::vector<float> &response,
				const float &sigma,
				const float &radius,
				const float &scale,
				const std::string &tag);
			void 	configure_loop_custom(const unsigned long long &id, const std::size_t &batch_size, const std::size_t &stimulus_size);

			void 	configure_scheduler_tiled(const unsigned long long &id ,const unsigned int &epochs);
			void 	configure_scheduler_snippets(const unsigned long long &id, const unsigned long &seed, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag);
	
			void 	configure_scheduler_custom(const unsigned long long &id, const unsigned long &seed, const std::string &tag);

			void 	configure_mutator_shuffle(const unsigned long long &id, const unsigned long &seed);
			void 	configure_mutator_reverse(const unsigned long long &id, const unsigned long &seed, const float &rate, const std::size_t &size);
			void 	configure_mutator_punch(const unsigned long long &id, const unsigned long &seed, const float &rate, const std::size_t &size, const std::size_t &number);

			void 	configure_mutator_custom(const unsigned long long &id, const unsigned long &seed);

			void 	configure_readout_uniform(const unsigned long long &id, const float &a, const float &b, const float &sparsity);
			void 	configure_readout_gaussian(const unsigned long long &id, const float &mu, const float &sigma);
			void 	configure_readout_custom(const unsigned long long &id);

			void 	configure_feedback_uniform(const unsigned long long &id, const float &a, const float &b, const float &sparsity);
			void 	configure_feedback_gaussian(const unsigned long long &id, const float &mu, const float &sigma);
			void 	configure_feedback_custom(const unsigned long long &id);

			void 	configure_recurrent_uniform(const unsigned long long &id, const float &a, const float &b, const float &sparsity);
			void 	configure_recurrent_gaussian(const unsigned long long &id, const float &mu, const float &sigma);
			void 	configure_recurrent_custom(const unsigned long long &id);

			void 	configure_feedforward_uniform(const unsigned long long &id, const float &a, const float &b, const float &sparsity);
			void 	configure_feedforward_gaussian(const unsigned long long &id ,const float &mu, const float &sigma);
			void 	configure_feedforward_custom(const unsigned long long &id);

	
		protected:
		
			virtual void callback_completed() = 0;
			virtual void callback_ack(const unsigned long long &id, const std::size_t &number, const bool &success, const std::string &cause) = 0;
			virtual void callback_processor(const int &rank, const std::string &host, const unsigned int &index, const std::string &name) = 0;
			virtual void callback_allocated(const unsigned long long &id, const int &rank) = 0;
			virtual void callback_deallocated(const unsigned long long &id, const int &rank) = 0;
			virtual void callback_quit(const int &rank) = 0;
			virtual void callback_configured(const unsigned long long &id) = 0;
			virtual void callback_trained(const unsigned long long &id) = 0;
			virtual void callback_primed(const unsigned long long &id) = 0;
			virtual void callback_tested(const unsigned long long &id) = 0;
			virtual void callback_error(const std::string &message) = 0;
			virtual void callback_information(const std::string &message) = 0;
			virtual void callback_warning(const std::string &message) = 0;


			virtual void callback_measurement_readout_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) = 0;
			virtual void callback_measurement_readout_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) = 0;
			virtual void callback_measurement_readout_custom(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) = 0;
			virtual void callback_measurement_position_mean_square_error(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) = 0;
			virtual void callback_measurement_position_frechet_distance(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &values, const std::size_t &rows, const  std::size_t &cols) = 0;
			virtual void callback_measurement_position_custom(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &primed, const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const  std::size_t &cols) = 0;

			virtual void callback_performances(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::string &phase, const float &cycles_per_second, const float &gflops_per_second) = 0;
			virtual void callback_states(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) = 0;
			virtual void callback_weights(const unsigned long long &id, const std::string &phase, const std::string &label, const std::size_t &batch, const std::size_t &trial, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols) = 0;
			virtual void callback_position(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols) = 0;
			virtual void callback_stimulus(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols) = 0;

			virtual void callback_mutator(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations) = 0;
			virtual void callback_scheduler(const unsigned long long &id, const unsigned long &seed, const std::size_t &trial, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations) = 0;
			virtual void callback_scheduling(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations) = 0;

			virtual void callback_feedforward(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) = 0;
			virtual void callback_feedback(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) = 0;
			virtual void callback_readout(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const  std::size_t &cols) = 0;
			virtual void callback_recurrent(const unsigned long long &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols) = 0;
		public :
			void notify_stimulus(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols);
			void notify_position(const unsigned long long &id, const std::size_t &trial, const std::size_t &evaluation, const std::vector<float> &estimated_position, const std::size_t &rows, const std::size_t &cols);
			void notify_scheduler(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations);
			void notify_mutator(const unsigned long long &id, const std::size_t &trial, const std::vector<int> &offsets, const std::vector<int> &durations);
			void notify_feedforward(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols);
			void notify_feedback(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols);
			void notify_readout(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols);
			void notify_recurrent(const unsigned long long &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols);
		private :
			std::size_t generate_number();
			template<TRN::Engine::Tag tag>
			void send(const int &rank, TRN::Engine::Message<tag> &message, const std::function<void()> &functor);
			void append_simulation(const unsigned long long &id);
			void remove_simulation(const unsigned long long &id);	
			std::shared_ptr<TRN::Engine::Executor> retrieve_simulation(const unsigned long long &id);
		};
	};
};
