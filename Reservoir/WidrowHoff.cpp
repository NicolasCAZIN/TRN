#include "stdafx.h"
#include "WidrowHoff_impl.h"
#include "Core/Reservoir_impl.h"

TRN::Reservoir::WidrowHoff::WidrowHoff(const std::shared_ptr<TRN::Backend::Driver> &driver,
							const std::size_t &stimulus, const std::size_t &prediction, const std::size_t &reservoir,
							const float &leak_rate,
							const float &initial_state_scale,
							const float &learning_rate,
							const unsigned long &seed,
							const std::size_t &batch_size, const std::size_t &mini_batch_size) :
	TRN::Core::Reservoir(driver, stimulus, prediction, reservoir, leak_rate, initial_state_scale, seed, batch_size, mini_batch_size),
	handle(std::make_unique<TRN::Reservoir::WidrowHoff::Handle >())
{
	std::vector<float> l = { learning_rate };
	handle->learning_rate = TRN::Core::Matrix::create(driver, l, 1, 1);
}

void TRN::Reservoir::WidrowHoff::train(const std::shared_ptr<TRN::Core::Matrix> &incoming, 
										const std::shared_ptr<TRN::Core::Matrix> &expected,
									   const std::shared_ptr<TRN::Core::Scheduling> &scheduling,
									   std::shared_ptr<TRN::Core::Matrix> &states)
{


		implementor->get_algorithm()->learn_widrow_hoff(
			TRN::Core::Reservoir::handle->batch_size, TRN::Core::Reservoir::handle->mini_batch_size,
			TRN::Core::Reservoir::handle->seed,
			TRN::Core::Reservoir::handle->stimulus_stride, TRN::Core::Reservoir::handle->reservoir_stride, TRN::Core::Reservoir::handle->prediction_stride,
			TRN::Core::Reservoir::handle->stimulus_size, TRN::Core::Reservoir::handle->reservoir_size, TRN::Core::Reservoir::handle->prediction_size,
			TRN::Core::Reservoir::handle->leak_rate, TRN::Core::Reservoir::handle->initial_state_scale,
			expected->get_elements(), expected->get_rows(), expected->get_cols(), expected->get_stride(),
		
			TRN::Core::Reservoir::handle->batched_incoming->get_elements(), TRN::Core::Reservoir::handle->batched_incoming->get_rows(), TRN::Core::Reservoir::handle->batched_incoming->get_cols(), TRN::Core::Reservoir::handle->batched_incoming->get_strides(),
			TRN::Core::Reservoir::handle->batched_expected->get_elements(), TRN::Core::Reservoir::handle->batched_expected->get_rows(), TRN::Core::Reservoir::handle->batched_expected->get_cols(), TRN::Core::Reservoir::handle->batched_expected->get_strides(),
			TRN::Core::Reservoir::handle->batched_W_ffwd->get_elements(), TRN::Core::Reservoir::handle->batched_W_ffwd->get_rows(), TRN::Core::Reservoir::handle->batched_W_ffwd->get_cols(), TRN::Core::Reservoir::handle->batched_W_ffwd->get_strides(),
			TRN::Core::Reservoir::handle->batched_u_ffwd->get_elements(), TRN::Core::Reservoir::handle->batched_u_ffwd->get_rows(), TRN::Core::Reservoir::handle->batched_u_ffwd->get_cols(), TRN::Core::Reservoir::handle->batched_u_ffwd->get_strides(),
			TRN::Core::Reservoir::handle->batched_X_res->get_elements(), TRN::Core::Reservoir::handle->batched_X_res->get_rows(), TRN::Core::Reservoir::handle->batched_X_res->get_cols(), TRN::Core::Reservoir::handle->batched_X_res->get_strides(),
			TRN::Core::Reservoir::handle->batched_W_rec->get_elements(), TRN::Core::Reservoir::handle->batched_W_rec->get_rows(), TRN::Core::Reservoir::handle->batched_W_rec->get_cols(), TRN::Core::Reservoir::handle->batched_W_rec->get_strides(),
			TRN::Core::Reservoir::handle->batched_u->get_elements(), TRN::Core::Reservoir::handle->batched_u->get_rows(), TRN::Core::Reservoir::handle->batched_u->get_cols(), TRN::Core::Reservoir::handle->batched_u->get_strides(),
			TRN::Core::Reservoir::handle->batched_p->get_elements(), TRN::Core::Reservoir::handle->batched_p->get_rows(), TRN::Core::Reservoir::handle->batched_p->get_cols(), TRN::Core::Reservoir::handle->batched_p->get_strides(),
			TRN::Core::Reservoir::handle->batched_X_ro->get_elements(), TRN::Core::Reservoir::handle->batched_X_ro->get_rows(), TRN::Core::Reservoir::handle->batched_X_ro->get_cols(), TRN::Core::Reservoir::handle->batched_X_ro->get_strides(),
			TRN::Core::Reservoir::handle->batched_W_ro->get_elements(), TRN::Core::Reservoir::handle->batched_W_ro->get_rows(), TRN::Core::Reservoir::handle->batched_W_ro->get_cols(), TRN::Core::Reservoir::handle->batched_W_ro->get_strides(),
			TRN::Core::Reservoir::handle->bundled_pre->get_elements(true), TRN::Core::Reservoir::handle->bundled_pre->get_rows(), TRN::Core::Reservoir::handle->bundled_pre->get_cols(), TRN::Core::Reservoir::handle->bundled_pre->get_strides(),
			TRN::Core::Reservoir::handle->batched_post->get_elements(), TRN::Core::Reservoir::handle->batched_post->get_rows(), TRN::Core::Reservoir::handle->batched_post->get_cols(), TRN::Core::Reservoir::handle->batched_post->get_strides(),
			TRN::Core::Reservoir::handle->bundled_desired->get_elements(true), TRN::Core::Reservoir::handle->bundled_desired->get_rows(), TRN::Core::Reservoir::handle->bundled_desired->get_cols(), TRN::Core::Reservoir::handle->bundled_desired->get_strides(),
			scheduling->get_offsets().data(), scheduling->get_durations().data(), scheduling->get_durations().size(), scheduling->get_total_duration(),
			states->get_elements(), states->get_rows(), states->get_cols(), states->get_stride(),
			 TRN::Core::Reservoir::handle->one->get_elements(), TRN::Core::Reservoir::handle->zero->get_elements(), handle->learning_rate->get_elements());
}


void TRN::Reservoir::WidrowHoff::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload)
{
	TRN::Core::Reservoir::visit(payload);

	auto flops_per_cycle = payload->get_flops_per_cycle();
	flops_per_cycle += TRN::Core::Reservoir::handle->reservoir_size * 2;
	flops_per_cycle += TRN::Core::Reservoir::handle->prediction_size * TRN::Core::Reservoir::handle->reservoir_size * 2;

	payload->set_flops_per_cycle(flops_per_cycle);
}

std::shared_ptr<TRN::Reservoir::WidrowHoff> TRN::Reservoir::WidrowHoff::create(const std::shared_ptr<TRN::Backend::Driver> &driver,
																				const std::size_t &stimulus, const std::size_t &prediction, const std::size_t &reservoir,
																				const float &leak_rate,
																				const float &initial_state_scale,
																				const float &learning_rate,
																				const unsigned long &seed,
																				const std::size_t &batch_size,
																				const std::size_t &mini_batch_size)
{
	return std::make_shared<TRN::Reservoir::WidrowHoff>(driver, stimulus, prediction, reservoir, leak_rate, initial_state_scale, learning_rate, seed, batch_size, mini_batch_size);
}
