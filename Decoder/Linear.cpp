#include "stdafx.h"
#include "Linear_impl.h"
#include "Core/Decoder_impl.h"

TRN::Decoder::Linear::Linear(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, const std::shared_ptr<TRN::Core::Matrix> &cx, const std::shared_ptr<TRN::Core::Matrix> &cy) :
	TRN::Core::Decoder(driver, batch_size, stimulus_size),
	handle(std::make_unique<Handle>())
{
	handle->cx = cx;
	handle->cy = cy;
}

TRN::Decoder::Linear::~Linear()
{
	handle.reset();
}

void TRN::Decoder::Linear::decode(
	const std::shared_ptr<TRN::Core::Batch> &previous_position,
	const std::shared_ptr<TRN::Core::Batch> &current_position,
	const std::shared_ptr<TRN::Core::Batch> &predicted_activations,
	std::shared_ptr<TRN::Core::Batch> &decoded_position)
{
	implementor->get_algorithm()->decode_placecells_linear
	(
		TRN::Core::Decoder::handle->batch_size,
		TRN::Core::Decoder::handle->stimulus_size,
		handle->cx->get_elements(),
		handle->cy->get_elements(),
		(const float **)predicted_activations->get_elements(true), predicted_activations->get_strides(),
		decoded_position->get_elements(true), decoded_position->get_strides()
	);
}

void TRN::Decoder::Linear::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload)
{
	payload->set_flops_per_epoch_factor(0);

	std::size_t flops_per_cycle = 0;
	auto place_cells = handle->cx->get_cols();
	flops_per_cycle += place_cells;
	flops_per_cycle += 1;
	flops_per_cycle += 2 * place_cells;

	payload->set_flops_per_cycle(flops_per_cycle);
}

std::shared_ptr<TRN::Decoder::Linear> TRN::Decoder::Linear::create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::size_t &batch_size, const std::size_t &stimulus_size, const std::shared_ptr<TRN::Core::Matrix> &cx, const std::shared_ptr<TRN::Core::Matrix> &cy)
{
	return std::make_shared<TRN::Decoder::Linear>(driver, batch_size, stimulus_size, cx, cy);
}
