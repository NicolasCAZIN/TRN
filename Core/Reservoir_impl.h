#pragma once

#include "Reservoir.h"
#include "Helper/Queue.h"
#include "Batch.h"

class TRN::Core::Reservoir::Handle
{
public:

	/*std::vector<std::shared_ptr<TRN::Core::Matrix>> _W_ro;
	std::vector<std::shared_ptr<TRN::Core::Matrix>> _W_rec;
	std::vector<std::shared_ptr<TRN::Core::Matrix>> _W_in;
	std::vector<std::shared_ptr<TRN::Core::Matrix>> _W_fbck;
	std::vector<std::shared_ptr<TRN::Core::Matrix>> _W_ffwd;*/
	std::shared_ptr<TRN::Core::Matrix> unitary_sub_u_ffwd;
	std::shared_ptr<TRN::Core::Batch> batched_incoming;
	std::shared_ptr<TRN::Core::Batch> batched_expected;
	std::shared_ptr<TRN::Core::Batch> batched_error;
	std::shared_ptr<TRN::Core::Batch> batched_u_ffwd;
	std::shared_ptr<TRN::Core::Batch> batched_u;
	std::shared_ptr<TRN::Core::Batch> batched_p;
	std::shared_ptr<TRN::Core::Batch> batched_W_ffwd;
	std::shared_ptr<TRN::Core::Batch> batched_X_in;
	std::shared_ptr<TRN::Core::Batch> batched_W_in;
	std::shared_ptr<TRN::Core::Batch> batched_W_rec;
	std::shared_ptr<TRN::Core::Batch> batched_W_fbck;
	std::shared_ptr<TRN::Core::Batch> batched_X_res;
	std::shared_ptr<TRN::Core::Batch> batched_X_ro;
	std::shared_ptr<TRN::Core::Batch> batched_W_ro;


	std::size_t stimulus_size;
	std::size_t reservoir_size;
	std::size_t prediction_size;
	std::size_t batch_size;
	std::size_t stimulus_stride;
	std::size_t reservoir_stride;
	std::size_t prediction_stride;
	
	unsigned long seed;
	float leak_rate;
	float initial_state_scale;

	bool gather_states;

	TRN::Helper::Queue<std::shared_ptr<TRN::Core::Batch>> prediction;
	
	std::mutex mutex;
	std::condition_variable end_of_test;
	std::thread thread;
	std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::STATES>> states;
	std::shared_ptr<TRN::Core::Matrix> target_expected;
	size_t cycle;
	size_t max_cycle;
};