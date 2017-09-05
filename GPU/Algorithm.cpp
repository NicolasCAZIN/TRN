#include "stdafx.h"

#include "Algorithm_impl.h"
#pragma warning(disable : 4068)
#include "Algorithm.cuh"


TRN::GPU::Algorithm::Algorithm(const std::shared_ptr<Context> context):
	handle(std::make_unique<TRN::GPU::Algorithm::Handle>())
{
	handle->context = context;
	checkCudaErrors(cudaMalloc(&handle->max_value, sizeof(float)));
	checkCudaErrors(cudaMalloc(&handle->argmax_value, 2 * sizeof(float)));
}

TRN::GPU::Algorithm::~Algorithm()
{
	checkCudaErrors(cudaStreamSynchronize(handle->context->get_stream()));
	checkCudaErrors(cudaFree(handle->max_value));
	checkCudaErrors(cudaFree(handle->argmax_value));

	handle.reset();
}

void TRN::GPU::Algorithm::preallocate(const std::size_t &stimulus_size, const std::size_t &reservoir_size,
	const std::size_t &prediction_size, const std::size_t &batch_size)
{
}

void TRN::GPU::Algorithm::mean_square_error
(
	const std::size_t &batch_size,
	const float **batched_predicted, const std::size_t *batched_predicted_rows, const std::size_t *batched_predicted_cols, const std::size_t *batched_predicted_strides,
	const float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float *result, const std::size_t &result_rows, const std::size_t &result_cols, const std::size_t &result_stride
)
{
	compute_mean_square_error
	(
		handle->context->get_stream(),
		handle->context->get_handle(),
		batch_size,
		batched_predicted, *batched_predicted_rows, *batched_predicted_cols, *batched_predicted_strides,
		expected, expected_rows, expected_cols, expected_stride,
		result, result_rows, result_cols, result_stride
	);
}




void TRN::GPU::Algorithm::place_cell_location_probability(
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float &sigma,
	const float ** firing_rate_map, const std::size_t *firing_rate_map_rows, const std::size_t *firing_rate_map_cols, const std::size_t *firing_rate_map_strides,
	float **scale, const std::size_t *scale_rows, const std::size_t *scale_cols, const std::size_t *scale_strides,
	const float **prediction, const std::size_t *prediction_rows, const std::size_t *prediction_cols, const std::size_t *prediction_strides,
	float *** hypothesis_map, const std::size_t **hypothesis_map_rows, const std::size_t **hypothesis_map_cols, const std::size_t **hypothesis_map_strides,
	float ** location_probability, const std::size_t *location_probability_rows, const std::size_t *location_probability_cols, const std::size_t *location_probability_strides)
{

	compute_place_cell_location_probability(handle->context->get_stream(), handle->context->get_handle(),
			batch_size, place_cells_number, rows, cols,
			sigma,
			firing_rate_map, *firing_rate_map_rows, *firing_rate_map_cols, *firing_rate_map_strides,
			scale, *scale_rows, *scale_cols, *scale_strides,
			prediction, *prediction_rows, *prediction_cols, *prediction_strides,
			hypothesis_map, **hypothesis_map_rows, **hypothesis_map_cols, **hypothesis_map_strides,
			location_probability, *location_probability_rows, *location_probability_cols, *location_probability_strides);


}


void TRN::GPU::Algorithm::restrict_to_reachable_locations
(
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float &radius, const float &scale,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float **batched_current_location, const std::size_t *batched_current_location_rows, const std::size_t *batched_current_location_cols, const std::size_t *batched_current_location_stride,
	 float **batched_x_grid_centered2, const std::size_t *batched_x_grid_centered2_rows, const std::size_t *batched_x_grid_centered2_cols, const std::size_t *batched_x_grid_centered2_stride,
	 float **batched_y_grid_centered2, const std::size_t *batched_y_grid_centered2_rows, const std::size_t *batched_y_grid_centered2_cols, const std::size_t *batched_y_grid_centered2_stride,
	float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides)
{
	compute_reachable_locations(handle->context->get_stream(), handle->context->get_handle(),batch_size, place_cells_number, rows, cols, radius, scale,
		x_grid, x_grid_rows, x_grid_cols, x_grid_stride,
		y_grid, y_grid_rows, y_grid_cols, y_grid_stride,
		batched_current_location, *batched_current_location_rows, *batched_current_location_cols, *batched_current_location_stride,
		batched_x_grid_centered2, *batched_x_grid_centered2_rows,*batched_x_grid_centered2_cols, *batched_x_grid_centered2_stride,
		batched_y_grid_centered2, *batched_y_grid_centered2_rows, *batched_y_grid_centered2_cols, *batched_y_grid_centered2_stride,
		batched_location_probability, *batched_location_probability_rows, *batched_location_probability_cols, *batched_location_probability_strides

		);
}


void TRN::GPU::Algorithm::draw_probable_location(const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides,
	float **batched_reduced_location_probability, const std::size_t *batched_reduced_location_probability_rows, const std::size_t *batched_reduced_location_probability_cols, const std::size_t *batched_reduced_location_probability_strides,
	float **batched_row_cumsum, const std::size_t *batched_row_cumsum_rows, const std::size_t *batched_row_cumsum_cols, const std::size_t *batched_row_cumsum_strides,
	float **batched_col_cumsum, const std::size_t *batched_col_cumsum_rows, const std::size_t *batched_col_cumsum_cols, const std::size_t *batched_col_cumsum_strides,
	float **batched_predicted_location, const std::size_t *batched_predicted_location_rows, const std::size_t *batched_predicted_location_cols, const std::size_t *batched_predicted_location_strides
)
{
	compute_draw_probable_location(handle->context->get_stream(), handle->context->get_handle(), batch_size, rows, cols,
		x_grid, x_grid_rows, x_grid_cols, x_grid_stride,
		y_grid, y_grid_rows, y_grid_cols, y_grid_stride,
		batched_location_probability, *batched_location_probability_rows, *batched_location_probability_cols, *batched_location_probability_strides,
		batched_reduced_location_probability, *batched_reduced_location_probability_rows, *batched_reduced_location_probability_cols, *batched_reduced_location_probability_strides,
		batched_row_cumsum, *batched_row_cumsum_rows, *batched_row_cumsum_cols,*batched_row_cumsum_strides,
		batched_col_cumsum, *batched_col_cumsum_rows, *batched_col_cumsum_cols, *batched_col_cumsum_strides,
		batched_predicted_location, *batched_predicted_location_rows, *batched_predicted_location_cols, *batched_predicted_location_strides
		);
 }

void TRN::GPU::Algorithm::select_most_probable_location(const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides,
	float **batched_predicted_location, const std::size_t *batched_predicted_location_rows, const std::size_t *batched_predicted_location_cols, const std::size_t *batched_predicted_location_strides
)
{
	compute_select_most_probable_location(handle->context->get_stream(), batch_size, rows, cols,
		x_grid, x_grid_rows, x_grid_cols, x_grid_stride,
		y_grid, y_grid_rows, y_grid_cols, y_grid_stride,
		batched_location_probability, *batched_location_probability_rows, *batched_location_probability_cols, *batched_location_probability_strides,
		batched_predicted_location, *batched_predicted_location_rows, *batched_predicted_location_cols, *batched_predicted_location_strides
	);
}
void TRN::GPU::Algorithm::learn_widrow_hoff(
	const std::size_t &batch_size,
	unsigned long &seed,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_w_ffwd, const std::size_t *batched_w_ffwd_rows, const std::size_t *batched_w_ffwd_cols, const std::size_t *batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const std::size_t *batched_u_ffwd_strides,
	float **batched_x_in, const std::size_t *batched_x_in_rows, const std::size_t *batched_x_in_cols, const std::size_t *batched_x_in_strides,
	float **batched_w_in, const std::size_t *batched_w_in_rows, const std::size_t *batched_w_in_cols, const std::size_t *batched_w_in_strides,
	float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const std::size_t *batched_u_strides,
	float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const std::size_t *batched_p_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,

	const float &learning_rate
	) 
{
	if (states_samples == NULL)
	{
		update_model<false, true>(
			batch_size,
			seed,
			handle->context->get_stream(), handle->context->get_handle(), Widrow_Hoff(learning_rate),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,	
			batched_incoming, *batched_incoming_rows, *batched_incoming_cols, *batched_incoming_strides,
			batched_expected, *batched_expected_rows, *batched_expected_cols, *batched_expected_strides,
			batched_w_ffwd, *batched_w_ffwd_rows, *batched_w_ffwd_cols, *batched_w_ffwd_strides,
			batched_u_ffwd, *batched_u_ffwd_rows, *batched_u_ffwd_cols, *batched_u_ffwd_strides,
			batched_x_in, *batched_x_in_rows, *batched_x_in_cols, *batched_x_in_strides,
			batched_w_in, *batched_w_in_rows, *batched_w_in_cols, *batched_w_in_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p,* batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_error, *batched_error_rows, *batched_error_cols, *batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride);
	}
	else
	{
		update_model<true, true>(
			batch_size,
			seed,
			handle->context->get_stream(), handle->context->get_handle(), Widrow_Hoff(learning_rate),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, *batched_incoming_rows, *batched_incoming_cols, *batched_incoming_strides,
			batched_expected, *batched_expected_rows, *batched_expected_cols, *batched_expected_strides,
			batched_w_ffwd, *batched_w_ffwd_rows, *batched_w_ffwd_cols, *batched_w_ffwd_strides,
			batched_u_ffwd, *batched_u_ffwd_rows, *batched_u_ffwd_cols, *batched_u_ffwd_strides,
			batched_x_in, *batched_x_in_rows, *batched_x_in_cols, *batched_x_in_strides,
			batched_w_in, *batched_w_in_rows, *batched_w_in_cols, *batched_w_in_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p, *batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_error, *batched_error_rows, *batched_error_cols, *batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride);
	}
}

void TRN::GPU::Algorithm::prime(
	const std::size_t &batch_size,
	unsigned long &seed,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_w_ffwd, const std::size_t *batched_w_ffwd_rows, const std::size_t *batched_w_ffwd_cols, const std::size_t *batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const std::size_t *batched_u_ffwd_strides,
	float **batched_x_in, const std::size_t *batched_x_in_rows, const std::size_t *batched_x_in_cols, const std::size_t *batched_x_in_strides,
	float **batched_w_in, const std::size_t *batched_w_in_rows, const std::size_t *batched_w_in_cols, const std::size_t *batched_w_in_strides,
	float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const std::size_t *batched_u_strides,
	float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const std::size_t *batched_p_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{
	if (states_samples == NULL)
	{
		update_model<false, true>(
			batch_size,
			seed,
				handle->context->get_stream(), handle->context->get_handle(), Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, *batched_incoming_rows, *batched_incoming_cols, *batched_incoming_strides,
			batched_expected, *batched_expected_rows, *batched_expected_cols, *batched_expected_strides,
			batched_w_ffwd, *batched_w_ffwd_rows, *batched_w_ffwd_cols, *batched_w_ffwd_strides,
			batched_u_ffwd, *batched_u_ffwd_rows, *batched_u_ffwd_cols, *batched_u_ffwd_strides,
			batched_x_in, *batched_x_in_rows, *batched_x_in_cols, *batched_x_in_strides,
			batched_w_in, *batched_w_in_rows, *batched_w_in_cols, *batched_w_in_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p, *batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_error, *batched_error_rows, *batched_error_cols, *batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride);
	}
	else
	{
		update_model< true, true>(
			batch_size,
			seed,
			handle->context->get_stream(), handle->context->get_handle(),  Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, *batched_incoming_rows, *batched_incoming_cols, *batched_incoming_strides,
			batched_expected, *batched_expected_rows, *batched_expected_cols, *batched_expected_strides,
			batched_w_ffwd, *batched_w_ffwd_rows, *batched_w_ffwd_cols, *batched_w_ffwd_strides,
			batched_u_ffwd, *batched_u_ffwd_rows, *batched_u_ffwd_cols, *batched_u_ffwd_strides,
			batched_x_in, *batched_x_in_rows, *batched_x_in_cols, *batched_x_in_strides,
			batched_w_in, *batched_w_in_rows, *batched_w_in_cols, *batched_w_in_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p, *batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_error, *batched_error_rows, *batched_error_cols, *batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride);
	}
		
}

void TRN::GPU::Algorithm::generate(
	const std::size_t &batch_size,
	unsigned long &seed,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_w_ffwd, const std::size_t *batched_w_ffwd_rows, const std::size_t *batched_w_ffwd_cols, const std::size_t *batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const std::size_t *batched_u_ffwd_strides,
	float **batched_x_in, const std::size_t *batched_x_in_rows, const std::size_t *batched_x_in_cols, const std::size_t *batched_x_in_strides,
	float **batched_w_in, const std::size_t *batched_w_in_rows, const std::size_t *batched_w_in_cols, const std::size_t *batched_w_in_strides,
	float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const std::size_t *batched_u_strides,
	float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const std::size_t *batched_p_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{
	if (states_samples == NULL)
	{
		update_model<false, false>(
			batch_size,
			seed,
				handle->context->get_stream(), handle->context->get_handle(), Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, *batched_incoming_rows, *batched_incoming_cols, *batched_incoming_strides,
			batched_expected, *batched_expected_rows, *batched_expected_cols, *batched_expected_strides,
			batched_w_ffwd, *batched_w_ffwd_rows, *batched_w_ffwd_cols, *batched_w_ffwd_strides,
			batched_u_ffwd, *batched_u_ffwd_rows, *batched_u_ffwd_cols, *batched_u_ffwd_strides,
			batched_x_in, *batched_x_in_rows, *batched_x_in_cols, *batched_x_in_strides,
			batched_w_in, *batched_w_in_rows, *batched_w_in_cols, *batched_w_in_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p, *batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_error, *batched_error_rows, *batched_error_cols, *batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride);
	}
	else
	{
		update_model<true, false>(
			batch_size,
			seed,
			handle->context->get_stream(), handle->context->get_handle(), Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,

			batched_incoming, *batched_incoming_rows, *batched_incoming_cols, *batched_incoming_strides,
			batched_expected, *batched_expected_rows, *batched_expected_cols, *batched_expected_strides,
			batched_w_ffwd, *batched_w_ffwd_rows, *batched_w_ffwd_cols, *batched_w_ffwd_strides,
			batched_u_ffwd, *batched_u_ffwd_rows, *batched_u_ffwd_cols, *batched_u_ffwd_strides,
			batched_x_in, *batched_x_in_rows, *batched_x_in_cols, *batched_x_in_strides,
			batched_w_in, *batched_w_in_rows, *batched_w_in_cols, *batched_w_in_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p, *batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_error, *batched_error_rows, *batched_error_cols, *batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride);
	}
}

std::shared_ptr<TRN::GPU::Algorithm> TRN::GPU::Algorithm::create(const std::shared_ptr<TRN::GPU::Context> context)
{
	return std::make_shared<TRN::GPU::Algorithm>(context);
}
