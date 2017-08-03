#pragma once

#include "engine_global.h"
#include "Messages.h"
#include "Manager.h"
#include "Communicator.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Broker
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Broker(const std::shared_ptr<TRN::Engine::Communicator> &communicator);
		public :
			~Broker();

		public :
			void start();
			void stop();

			void    setup_processor(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &processor);
			void    setup_allocation(const std::function<void(const unsigned int &id, const int &rank)> &on_allocation);
			void    setup_deallocation(const std::function<void(const unsigned int &id, const int &rank)> &on_deallocation);

			void 	allocate(const unsigned int &id);

			void 	deallocate(const unsigned int &id);
			void 	train(const unsigned int &id, const std::string &label, const std::string &incoming, const std::string &expected);
			void 	test(const unsigned int &id, const std::string &sequence, const std::string &incoming, const std::string &expected, const unsigned int &preamble);
			void 	declare_sequence(const unsigned int &id, const std::string &label, const std::string &tag,
				const std::vector<float> &sequence, const std::size_t &observations);
			void 	declare_set(const unsigned int &id, const std::string &label, const std::string &tag,
				const std::vector<std::string> &labels);
			void 	setup_states(const unsigned int &id, const std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &train, const bool &prime, const bool &generate);
			void 	setup_weights(const unsigned int &id, const std::function<void(const std::string &phase, const std::string &label, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initilization, const bool &train);
			void 	setup_performances(const unsigned int &id, const std::function<void(const std::string &phase, const size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &functor, const bool &train, const bool &prime, const bool &generate);

			void	configure_begin(const unsigned int &id);
			void	configure_end(const unsigned int &id);

			void 	configure_measurement_readout_mean_square_error(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
			void  	configure_measurement_readout_frechet_distance(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::vector<float> &values,  const std::size_t &rows, const std::size_t &cols)> &functor);
			void  	configure_measurement_readout_custom(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);

			void  	configure_measurement_position_mean_square_error(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
			void  	configure_measurement_position_frechet_distance(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
			void  	configure_measurement_position_custom(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const std::vector<float> &predicted, const std::vector<float> &expected, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);

			void 	configure_reservoir_widrow_hoff(const unsigned int &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate,
				const float &initial_state_scale, const float &learning_rate, const unsigned long &seed, const std::size_t &batch_size);
			/*virtual void 	configure_reservoir_online_force(const unsigned int &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate,
			const float &initial_state_scale, const float &learning_rate) = 0;*/
			/*	virtual void 	configure_reservoir_offline_svd(const unsigned int &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate,
			const float &initial_state_scale, const float &learning_rate) = 0;*/
			void 	configure_loop_copy(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size);
			void 	configure_loop_spatial_filter(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
				std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
				const std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
				std::function<void(const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
				const std::size_t &rows, const std::size_t &cols,
				const std::pair<float, float> &x, const std::pair<float, float> &y,
				const std::vector<float> &response,
				const float &sigma,
				const float &radius,
				const std::string &tag);
			void 	configure_loop_custom(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size,
				const std::function<void(const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply
			);

			void 	configure_scheduler_tiled(const unsigned int &id ,const unsigned int &epochs);
			void 	configure_scheduler_snippets(const unsigned int &id, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag);
			void 	configure_scheduler_custom(const unsigned int &id,
				const std::function<void(const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)> &request,
				std::function<void(const std::vector<unsigned int> &offsets, const std::vector<unsigned int> &durations)> &reply, const std::string &tag);

			void 	configure_readout_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity);
			void 	configure_readout_gaussian(const unsigned int &id, const float &mu, const float &sigma);
			void 	configure_readout_custom(const unsigned int &id,
				const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply
			);

			void 	configure_feedback_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity);
			void 	configure_feedback_gaussian(const unsigned int &id, const float &mu, const float &sigma);
			void 	configure_feedback_custom(const unsigned int &id,
				const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply
			);

			void 	configure_recurrent_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity);
			void 	configure_recurrent_gaussian(const unsigned int &id, const float &mu, const float &sigma);
			void 	configure_recurrent_custom(const unsigned int &id,
				const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply
			);

			void 	configure_feedforward_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity);
			void 	configure_feedforward_gaussian(const unsigned int &id ,const float &mu, const float &sigma);
			void 	configure_feedforward_custom(const unsigned int &id,
				const std::function<void(const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
				std::function<void(const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply
			);

	

		private :
		
			void receive();

			template<TRN::Engine::Tag tag>
			void send(const int &rank, TRN::Engine::Message<tag> &message, const std::function<void()> &functor);





		public :
			static std::shared_ptr<TRN::Engine::Broker> create(const std::shared_ptr<TRN::Engine::Communicator> &communicator);

		};
	};
};
