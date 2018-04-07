#pragma once

#include "Reservoir.h"
#include "Helper/Queue.h"

#include "Bundle.h"

class TRN::Core::Reservoir::Handle
{
public:
	std::shared_ptr<TRN::Core::Matrix> one;
	std::shared_ptr<TRN::Core::Matrix> zero;
	std::shared_ptr<TRN::Core::Matrix> unitary_sub_u_ffwd;
	std::shared_ptr<TRN::Core::Batch> batched_incoming;
	std::shared_ptr<TRN::Core::Batch> batched_expected;

	std::shared_ptr<TRN::Core::Batch> batched_u_ffwd;
	std::shared_ptr<TRN::Core::Batch> batched_u;
	std::shared_ptr<TRN::Core::Batch> batched_p;
	std::shared_ptr<TRN::Core::Batch> batched_W_ffwd;
	
	std::shared_ptr<TRN::Core::Batch> batched_W_rec;
	std::shared_ptr<TRN::Core::Batch> batched_X_res;
	std::shared_ptr<TRN::Core::Batch> batched_X_ro;
	std::shared_ptr<TRN::Core::Batch> batched_W_ro;
	std::shared_ptr<TRN::Core::Batch> batched_W_ro_reset;

	std::shared_ptr<TRN::Core::Batch> batched_post;
	std::shared_ptr<TRN::Core::Bundle> bundled_pre;
	std::shared_ptr<TRN::Core::Bundle> bundled_desired;

	std::size_t stimulus_size;
	std::size_t reservoir_size;
	std::size_t prediction_size;
	std::size_t batch_size;
	std::size_t mini_batch_size;
	std::size_t stimulus_stride;
	std::size_t reservoir_stride;
	std::size_t prediction_stride;
	
	unsigned long seed;
	float leak_rate;
	float initial_state_scale;
	std::thread thread;
	bool gather_states;
	bool autonomous_generation;
	TRN::Helper::Queue<std::tuple<std::shared_ptr<TRN::Core::Batch>, unsigned long long> > prediction;
	
	std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::STATES>> states;
	std::shared_ptr<TRN::Core::Matrix> target_expected;
	size_t cycle;
	size_t max_cycle;
};