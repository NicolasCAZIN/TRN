#include "stdafx.h"
#include "Map_impl.h"
#include "Kernel_impl.h"
#include "Core/Decoder_impl.h"
TRN::Decoder::Map::Map(const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &arena_x, const std::pair<float, float> &arena_y,
	const float &sigma, const float &radius, const float &angle, const float &scale, const unsigned long &seed,
	const std::shared_ptr<TRN::Core::Matrix> &firing_rate_map) :
	TRN::Decoder::Kernel(driver, batch_size, stimulus_size, rows, cols, arena_x, arena_y, sigma, radius, angle, scale, seed),
	handle(std::make_unique<Handle>())
{
	
	handle->firing_rate_map = firing_rate_map;
}

TRN::Decoder::Map::~Map()
{
	handle.reset();
}



void TRN::Decoder::Map::location_probability(
	const std::shared_ptr<TRN::Core::Batch> &previous_position,
	const std::shared_ptr<TRN::Core::Batch> &current_position,
	const std::shared_ptr<TRN::Core::Batch> &predicted_activations,
	std::shared_ptr<TRN::Core::Batch> &location_probability)
{
	implementor->get_algorithm()->decode_placecells_kernel_map
	(
		TRN::Core::Decoder::handle->batch_size,
		TRN::Core::Decoder::handle->stimulus_size,
		TRN::Decoder::Kernel::handle->rows,
		TRN::Decoder::Kernel::handle->cols,
		TRN::Decoder::Kernel::handle->roi_rows, TRN::Decoder::Kernel::handle->roi_cols,
		TRN::Decoder::Kernel::handle->roi_row_begin, TRN::Decoder::Kernel::handle->roi_row_end,
		TRN::Decoder::Kernel::handle->roi_col_begin, TRN::Decoder::Kernel::handle->roi_col_end,
		TRN::Decoder::Kernel::handle->arena_x.first, TRN::Decoder::Kernel::handle->arena_x.second,
		TRN::Decoder::Kernel::handle->arena_y.first, TRN::Decoder::Kernel::handle->arena_y.second,
		TRN::Decoder::Kernel::handle->radius,
		TRN::Decoder::Kernel::handle->cos_half_angle,
		TRN::Decoder::Kernel::handle->scale,
		TRN::Decoder::Kernel::handle->sigma,
		TRN::Decoder::Kernel::handle->seed,
		(const float *)handle->firing_rate_map->get_elements(), handle->firing_rate_map->get_stride(),
		TRN::Decoder::Kernel::handle->x_grid->get_elements(), TRN::Decoder::Kernel::handle->x_grid->get_stride(),
		TRN::Decoder::Kernel::handle->y_grid->get_elements(), TRN::Decoder::Kernel::handle->y_grid->get_stride(),
		(const float **)previous_position->get_elements(), previous_position->get_strides(),
		(const float **)current_position->get_elements(), current_position->get_strides(),
		(const float **)predicted_activations->get_elements(), predicted_activations->get_strides(),
		TRN::Decoder::Kernel::handle->batched_direction->get_elements(), TRN::Decoder::Kernel::handle->batched_direction->get_strides(),
		TRN::Decoder::Kernel::handle->batched_x_grid_centered->get_elements(), TRN::Decoder::Kernel::handle->batched_x_grid_centered->get_strides(),
		TRN::Decoder::Kernel::handle->batched_y_grid_centered->get_elements(), TRN::Decoder::Kernel::handle->batched_y_grid_centered->get_strides(),
		location_probability->get_elements(), location_probability->get_strides()
	);
}


void TRN::Decoder::Map::visit(std::shared_ptr<TRN::Core::Message::Payload<TRN::Core::Message::FLOPS>> &payload)
{
	payload->set_flops_per_epoch_factor(0);

	std::size_t flops_per_cycle = 0;
	auto place_cells = handle->firing_rate_map->get_cols();
	auto rows = TRN::Decoder::Kernel::handle->rows;
	auto cols = TRN::Decoder::Kernel::handle->cols;
	auto roi_rows = TRN::Decoder::Kernel::handle->roi_rows;
	auto roi_cols = TRN::Decoder::Kernel::handle->roi_cols;
	flops_per_cycle += rows;
	flops_per_cycle += cols;

	flops_per_cycle += place_cells * (roi_rows * roi_cols); // mul
	flops_per_cycle += place_cells * (roi_rows * roi_cols); // add
	flops_per_cycle += (roi_rows * roi_cols); // div
	flops_per_cycle += (roi_rows * roi_cols) * 32; // exp
	flops_per_cycle += (roi_rows * roi_cols);// noise
	flops_per_cycle += 3; // <a,a>
	flops_per_cycle += roi_rows * (1 + 1 + roi_cols* (3));
	//flops_per_cycle += (rows * cols);// argmax

	payload->set_flops_per_cycle(flops_per_cycle);
}

std::shared_ptr<TRN::Decoder::Map> TRN::Decoder::Map::create(const std::shared_ptr<TRN::Backend::Driver> &driver,
	const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::size_t &rows, const std::size_t &cols,
	const std::pair<float, float> &arena_x, const std::pair<float, float> &arena_y,
	const float &sigma, const float &radius, const float &angle, const float &scale, const unsigned long &seed,
	const std::shared_ptr<TRN::Core::Matrix> &firing_rate_map)
{
	return std::make_shared<TRN::Decoder::Map>(driver, batch_size, stimulus_size, rows, cols, arena_x, arena_y, sigma, radius, angle, scale, seed, firing_rate_map);
}


