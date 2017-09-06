#pragma once

#include "trn4cpp_global.h"

namespace TRN4CPP
{
	extern TRN4CPP_EXPORT const bool DEFAULT_INITIALIZE;
	extern TRN4CPP_EXPORT const bool DEFAULT_TRAIN;
	extern TRN4CPP_EXPORT const bool DEFAULT_PRIME;
	extern TRN4CPP_EXPORT const bool DEFAULT_GENERATE;
	extern TRN4CPP_EXPORT const size_t DEFAULT_BATCH_SIZE;
	extern TRN4CPP_EXPORT const unsigned long DEFAULT_SEED;
	extern TRN4CPP_EXPORT const unsigned int DEFAULT_INDEX;
	extern TRN4CPP_EXPORT const unsigned int DEFAULT_SUPPLEMENTARY_GENERATIONS;
	extern TRN4CPP_EXPORT const unsigned long DEFAULT_SEED;
	extern TRN4CPP_EXPORT const std::string DEFAULT_HOST;
	extern TRN4CPP_EXPORT const unsigned short DEFAULT_PORT;
	extern TRN4CPP_EXPORT const std::string DEFAULT_TAG;

	void TRN4CPP_EXPORT		install_processor(const std::function<void(const int &rank, const std::string &host, const unsigned int &index, const std::string &name)> &functor);
	void TRN4CPP_EXPORT		install_allocation(const std::function<void(const unsigned int &id, const int &rank)> &functor);
	void TRN4CPP_EXPORT		install_deallocation(const std::function<void(const unsigned int &id, const int &rank)> &functor);
	void TRN4CPP_EXPORT  	initialize_local(const std::list<unsigned int> &indexes = {});
	void TRN4CPP_EXPORT  	initialize_remote(const std::string &host = DEFAULT_HOST, const unsigned short &port = DEFAULT_PORT);
	void TRN4CPP_EXPORT  	initialize_distributed(int argc, char *argv[]);
	void TRN4CPP_EXPORT  	uninitialize();
	void TRN4CPP_EXPORT  	allocate(const unsigned int &id);
	void TRN4CPP_EXPORT  	deallocate(const unsigned int &id);
	
	void TRN4CPP_EXPORT  	train(const unsigned int &id, const std::string &label, const std::string &incoming, const std::string &expected);
	void TRN4CPP_EXPORT  	test(const unsigned int &id, const std::string &sequence,const std::string &incoming, const std::string &expected, const unsigned int &preamble, const unsigned int &supplementary_generations = 0);
	
	void TRN4CPP_EXPORT  	declare_sequence(const unsigned int &id, const std::string &label, const std::string &tag, const std::vector<float> &sequence, const std::size_t &observations);
	void TRN4CPP_EXPORT  	declare_set(const unsigned int &id, const std::string &label, const std::string &tag, const std::vector<std::string> &labels);
	
	void TRN4CPP_EXPORT  	setup_states(const unsigned int &id, const std::function<void (const unsigned int &id, const std::string &phase, const std::string &label, const std::vector<float> &samples, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &train = DEFAULT_TRAIN, const bool &prime = DEFAULT_PRIME, const bool &generate = DEFAULT_GENERATE);
	void TRN4CPP_EXPORT  	setup_weights(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::string &label, const std::vector<float> &weights, const std::size_t &rows, const std::size_t &cols)> &functor, const bool &initialize = DEFAULT_INITIALIZE, const bool &train = DEFAULT_TRAIN);
	void TRN4CPP_EXPORT  	setup_performances(const unsigned int &id, const std::function<void(const unsigned int &id, const std::string &phase, const std::size_t &batch_size, const size_t &cycles, const float &gflops, const float &seconds)> &functor, const bool &train = DEFAULT_TRAIN, const bool &prime = DEFAULT_PRIME, const bool &generate = DEFAULT_GENERATE);
	void TRN4CPP_EXPORT  	setup_scheduling(const unsigned int &id, const std::function<void(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)> &functor);

	void TRN4CPP_EXPORT 	configure_begin(const unsigned int &id);
	void TRN4CPP_EXPORT 	configure_end(const unsigned int &id);

	void TRN4CPP_EXPORT 	configure_measurement_readout_mean_square_error(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
	void TRN4CPP_EXPORT 	configure_measurement_readout_frechet_distance(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
	void TRN4CPP_EXPORT 	configure_measurement_readout_custom(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);

	void TRN4CPP_EXPORT 	configure_measurement_position_mean_square_error(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
	void TRN4CPP_EXPORT 	configure_measurement_position_frechet_distance(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &values, const std::size_t &rows, const std::size_t &cols)> &functor);
	void TRN4CPP_EXPORT 	configure_measurement_position_custom(const unsigned int &id, const std::size_t &batch_size, const std::function<void(const unsigned int &id, const std::vector<float> &predicted, const std::vector<float> &primed, const std::vector<float> &expected, const std::size_t &preamble, const std::size_t &pages, const std::size_t &rows, const std::size_t &cols)> &functor);



	void TRN4CPP_EXPORT  	configure_reservoir_widrow_hoff(const unsigned int &id, const std::size_t &stimulus_size, const std::size_t &prediction_size, const std::size_t &reservoir_size, const float &leak_rate, const float &initial_state_scale, const float &learning_rate, const unsigned long &seed = DEFAULT_SEED, const std::size_t &batch_size = DEFAULT_BATCH_SIZE);

	void TRN4CPP_EXPORT  	configure_loop_copy(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size);
	void TRN4CPP_EXPORT  	configure_loop_spatial_filter(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size,
															const std::function<void(const unsigned int &id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_position,
															std::function<void(const unsigned int &id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &estimated_position,
															const std::function<void(const unsigned int &id, const std::vector<float> &position, const std::size_t &rows, const std::size_t &cols)> &predicted_stimulus,
															std::function<void(const unsigned int &id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &perceived_stimulus,
														    const std::size_t &rows, const std::size_t &cols, const std::pair<float, float> &x, const std::pair<float, float> &y,
															const std::vector<float> response, const float &sigma, const float &radius, const float &scale, const std::string &tag);
	void TRN4CPP_EXPORT  	configure_loop_custom(const unsigned int &id, const std::size_t &batch_size, const std::size_t &stimulus_size,
		const std::function<void(const unsigned int &id, const std::vector<float> &prediction, const std::size_t &rows, const std::size_t &cols)> &request,
		std::function<void(const unsigned int &id, const std::vector<float> &stimulus, const std::size_t &rows, const std::size_t &cols)> &reply
		);

	void TRN4CPP_EXPORT  	configure_scheduler_tiled(const unsigned int &id, const unsigned int &epochs);
	void TRN4CPP_EXPORT  	configure_scheduler_snippets(const unsigned int &id, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag = TRN4CPP::DEFAULT_TAG);
	/*void TRN4CPP_EXPORT  	configure_scheduler_snippets_shuffle(const unsigned int &id, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag = TRN4CPP::DEFAULT_TAG);
	void TRN4CPP_EXPORT  	configure_scheduler_snippets_repeat(const unsigned int &id, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag = TRN4CPP::DEFAULT_TAG);
	void TRN4CPP_EXPORT  	configure_scheduler_snippets_noise(const unsigned int &id, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag = TRN4CPP::DEFAULT_TAG);
	void TRN4CPP_EXPORT  	configure_scheduler_snippets_jitter(const unsigned int &id, const unsigned int &snippets_size, const unsigned int &time_budget, const std::string &tag = TRN4CPP::DEFAULT_TAG);
	*/
	void TRN4CPP_EXPORT  	configure_scheduler_custom(const unsigned int &id,
		const std::function<void(const unsigned int &id, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
		std::function<void(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply, const std::string &tag);

	void TRN4CPP_EXPORT  	configure_mutator_shuffle(const unsigned int &id);
	void TRN4CPP_EXPORT  	configure_mutator_reverse(const unsigned int &id, const float &rate, const std::size_t &size);
	void TRN4CPP_EXPORT  	configure_mutator_punch(const unsigned int &id, const float &rate,  const std::size_t &size, const std::size_t &number);
	void TRN4CPP_EXPORT  	configure_mutator_custom(const unsigned int &id,  
		const std::function<void(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)> &request,
		std::function<void(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)> &reply
		);

	void TRN4CPP_EXPORT  	configure_readout_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity);
	void TRN4CPP_EXPORT  	configure_readout_gaussian(const unsigned int &id, const float &mu, const float &sigma);
	void TRN4CPP_EXPORT  	configure_readout_custom(const unsigned int &id,
		const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
		std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);

	void TRN4CPP_EXPORT  	configure_feedback_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity);
	void TRN4CPP_EXPORT  	configure_feedback_gaussian(const unsigned int &id, const float &mu, const float &sigma);
	void TRN4CPP_EXPORT  	configure_feedback_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
		std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);

	void TRN4CPP_EXPORT  	configure_recurrent_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity);
	void TRN4CPP_EXPORT  	configure_recurrent_gaussian(const unsigned int &id, const float &mu, const float &sigma);
	void TRN4CPP_EXPORT  	configure_recurrent_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
		std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);

	void TRN4CPP_EXPORT  	configure_feedforward_uniform(const unsigned int &id, const float &a, const float &b, const float &sparsity);
	void TRN4CPP_EXPORT  	configure_feedforward_gaussian(const unsigned int &id, const float &mu, const float &sigma);
	void TRN4CPP_EXPORT  	configure_feedforward_custom(const unsigned int &id, const std::function<void(const unsigned int &id, const unsigned long &seed, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &request,
		std::function<void(const unsigned int &id, const std::vector<float> &weights, const std::size_t &matrices, const std::size_t &rows, const std::size_t &cols)> &reply);
};
