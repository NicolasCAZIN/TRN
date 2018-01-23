#include "stdafx.h"
#include "Copy.h"
#include "Core/Loop_impl.h"

#include <iostream>

TRN::Loop::Copy::Copy(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size) :
	TRN::Core::Loop(driver, batch_size, stimulus_size)
{
}

void TRN::Loop::Copy::update(const TRN::Core::Message::Payload<TRN::Core::Message::PREDICTION> &payload)
{
	/*if (prediction.get_rows() != stimulus.get_rows())
		throw std::invalid_argument("Prediction and Stimulus row number must be the same");
	if (prediction.get_cols() != stimulus.get_cols())
		throw std::invalid_argument("Prediction and Stimulus column number must be the same");*/
	auto predicted = payload.get_predicted();
	auto evaluation_id = payload.get_evaluation_id();
	for (std::size_t batch = 0; batch < handle->batch_size; batch++)
		handle->stimulus->get_matrices(batch)->from(*payload.get_predicted()->get_matrices(batch));
	
	TRN::Helper::Observable<TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>>::notify(TRN::Core::Message::Payload<TRN::Core::Message::STIMULUS>(handle->stimulus, evaluation_id));
}

void TRN::Loop::Copy::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload)
{
	payload->set_flops_per_cycle(0);
	payload->set_flops_per_epoch_factor(0);
}

std::shared_ptr<TRN::Loop::Copy> TRN::Loop::Copy::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size)
{
	return std::make_shared<TRN::Loop::Copy>(driver, batch_size, stimulus_size);
}

