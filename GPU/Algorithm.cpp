#include "stdafx.h"

#include "Algorithm_impl.h"
#pragma warning(disable : 4068)
#include "Algorithm.cuh"



TRN::GPU::Algorithm::Algorithm(const std::shared_ptr<Context> context):
	handle(std::make_unique<TRN::GPU::Algorithm::Handle>())
{
	handle->context = context;
	/*checkCudaErrors(cudaMalloc(&handle->max_value, sizeof(float)));
	checkCudaErrors(cudaMalloc(&handle->argmax_value, 2 * sizeof(float)));*/
}

TRN::GPU::Algorithm::~Algorithm()
{
	/*checkCudaErrors(cudaStreamSynchronize(handle->context->get_stream()));
	checkCudaErrors(cudaFree(handle->max_value));
	checkCudaErrors(cudaFree(handle->argmax_value));*/

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

void  TRN::GPU::Algorithm::compute_roi(const std::size_t &batch_size,
	const std::size_t &rows, const std::size_t &cols,
	const float &x_min, const float &x_max, const float &y_min, const float &y_max,
	const float &radius,
	const float **current_position, const std::size_t *current_position_strides,
	std::size_t *roi_row_begin, std::size_t *roi_row_end, std::size_t *roi_col_begin, std::size_t *roi_col_end)
{

}

void TRN::GPU::Algorithm::decode_placecells_linear(
	const std::size_t &batch_size, const std::size_t &place_cells_number,
	const float *cx,
	const float *cy,
	const float **batched_prediction, const std::size_t *batched_prediction_strides,
	float **batched_decoded_position, const std::size_t *batched_decoded_position_strides)
{
	compute_decode_placecells_linear(handle->context->get_stream(),
		handle->context->get_handle(), batch_size, place_cells_number, cx, cy, batched_prediction, *batched_prediction_strides, batched_decoded_position, *batched_decoded_position_strides);
}
void TRN::GPU::Algorithm::decode_placecells_kernel_model
(
	const std::size_t &batch_size, const std::size_t &place_cells_number,
	const std::size_t &rows, const std::size_t &cols,
	const std::size_t &roi_rows, const std::size_t &roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const float &radius,
	const float &cos_half_angle,
	const float &scale,
	const float &sigma,
	const unsigned long &seed,
	const float *cx, const float *cy, const float *width,
	float **gx2w, const std::size_t *gx2w_strides,
	float **gy2w, const std::size_t *gy2w_strides,
	const float *x_grid, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_stride,
	const float **batched_previous_position, const std::size_t *batched_previous_position_stride,
	const float **batched_current_position, const std::size_t *batched_current_position_stride,
	const float **batched_predicted_activations, const std::size_t *batched_predicted_activations_stride,
	float **batched_direction, const std::size_t *batched_direction_stride,
	float **batched_x_grid_centered, const std::size_t *batched_x_grid_centered_stride,
	float **batched_y_grid_centered, const std::size_t *batched_y_grid_centered_stride,
	float **batched_location_probability, const std::size_t *batched_location_probability_strides
)
{

}
void TRN::GPU::Algorithm::decode_placecells_kernel_map
(
	const std::size_t &batch_size, const std::size_t &place_cells_number,
	const std::size_t &rows, const std::size_t &cols,
	const std::size_t &roi_rows, const std::size_t &roi_cols,
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const float &radius,
	const float &cos_half_angle,
	const float &scale,
	const float &sigma,
	const unsigned long &seed,
	const float *firing_rate_map, const std::size_t &firing_rate_maps_stride,
	const float *x_grid, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_stride,
	const float **batched_previous_position, const std::size_t *batched_previous_position_stride,
	const float **batched_current_position, const std::size_t *batched_current_position_stride,
	const float **batched_predicted_activations, const std::size_t *batched_predicted_activations_stride,
	float **batched_direction, const std::size_t *batched_direction_stride,
	float **batched_x_grid_centered, const std::size_t *batched_x_grid_centered_stride,
	float **batched_y_grid_centered, const std::size_t *batched_y_grid_centered_stride,
	float **batched_location_probability, const std::size_t *batched_location_probability_strides
)
{

}

void TRN::GPU::Algorithm::decode_most_probable_location(
	const std::size_t &batch_size, const std::size_t &stimulus_size,
	const std::size_t &roi_row_begin, const std::size_t &roi_row_end,
	const std::size_t &roi_col_begin, const std::size_t &roi_col_end,
	const std::size_t &order,
	const unsigned long &seed, const float &sigma, const float &scale, const float &radius, const float &cos_half_angle,
	const float *firing_rate_map, const std::size_t &firing_rate_map_stride,
	const float *coefficients, const std::size_t &coefficients_stride,
	const float *x_grid, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_stride,
	const float **batched_previous_position, const std::size_t *batched_previous_position_strides,
	const float **batched_current_position, const std::size_t *batched_current_position_strides,
	const float **batched_prediction, const std::size_t *batched_prediction_strides,
	float *hypothesis_scale, const std::size_t &hypothesis_scale_stride,
	float **batched_x_grid_centered, const std::size_t *batched_x_grid_centered_strides,
	float **batched_y_grid_centered, const std::size_t *batched_y_grid_centered_strides,
	float **batched_direction, const std::size_t *batched_direction_strides,
	float **batched_decoded_position, const std::size_t *batched_decoded_position_strides
)
{
	compute_decode_most_probable_location
	(
		handle->context->get_stream(), handle->context->get_handle(),
		batch_size, stimulus_size,
		roi_row_begin, roi_row_end,
		roi_col_begin, roi_col_end,
		order,
		seed, sigma, scale, radius, cos_half_angle,
		firing_rate_map, firing_rate_map_stride,
		coefficients, coefficients_stride,
		x_grid, x_grid_stride,
		y_grid, y_grid_stride,
		batched_previous_position, *batched_previous_position_strides,
		batched_current_position, *batched_current_position_strides,
		batched_prediction, *batched_prediction_strides,
		hypothesis_scale, hypothesis_scale_stride,
		batched_x_grid_centered, *batched_x_grid_centered_strides,
		batched_y_grid_centered, *batched_y_grid_centered_strides,
		batched_direction, *batched_direction_strides,
		batched_decoded_position, *batched_decoded_position_strides
	);
}

void TRN::GPU::Algorithm::place_cell_location_probability(
	const std::size_t &batch_size, const std::size_t &place_cells_number, 
	const std::size_t &rows_begin, const std::size_t &rows_end,
	const std::size_t &cols_begin, const std::size_t &cols_end,
	const float &sigma,
	const float ** firing_rate_map, const std::size_t *firing_rate_map_rows, const std::size_t *firing_rate_map_cols, const std::size_t *firing_rate_map_strides,
	float **scale, const std::size_t *scale_rows, const std::size_t *scale_cols, const std::size_t *scale_strides,
	const float **prediction, const std::size_t *prediction_rows, const std::size_t *prediction_cols, const std::size_t *prediction_strides,
	float *** hypothesis_map, const std::size_t **hypothesis_map_rows, const std::size_t **hypothesis_map_cols, const std::size_t **hypothesis_map_strides,
	float ** location_probability, const std::size_t *location_probability_rows, const std::size_t *location_probability_cols, const std::size_t *location_probability_strides)
{

	compute_place_cell_location_probability(handle->context->get_stream(), handle->context->get_handle(),
			batch_size, place_cells_number, 
			rows_begin, rows_end, 
			cols_begin, cols_end,
			sigma,
			firing_rate_map, *firing_rate_map_rows, *firing_rate_map_cols, *firing_rate_map_strides,
			scale, *scale_rows, *scale_cols, *scale_strides,
			prediction, *prediction_rows, *prediction_cols, *prediction_strides,
			hypothesis_map, **hypothesis_map_rows, **hypothesis_map_cols, **hypothesis_map_strides,
			location_probability, *location_probability_rows, *location_probability_cols, *location_probability_strides);


}


void TRN::GPU::Algorithm::restrict_to_reachable_locations
(
	const std::size_t &batch_size, const std::size_t &place_cells_number, 
	
	const std::size_t &rows_begin, const std::size_t &rows_end,
	const std::size_t &cols_begin, const std::size_t &cols_end,
	const float &radius, const float &cos_half_angle, const float &scale, const unsigned long &seed,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float **batched_previous_location, const std::size_t *batched_previous_location_rows, const std::size_t *batched_previous_location_cols, const std::size_t *batched_previous_location_stride,
	const float **batched_current_location, const std::size_t *batched_current_location_rows, const std::size_t *batched_current_location_cols, const std::size_t *batched_current_location_stride,
	float **batched_direction, const std::size_t *batched_direction_rows, const std::size_t *batched_direction_cols, const std::size_t *batched_direction_stride,
	float **batched_x_grid_centered, const std::size_t *batched_x_grid_centered_rows, const std::size_t *batched_x_grid_centered_cols, const std::size_t *batched_x_grid_centered_stride,
	float **batched_y_grid_centered, const std::size_t *batched_y_grid_centered_rows, const std::size_t *batched_y_grid_centered_cols, const std::size_t *batched_y_grid_centered_stride,
	float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides)
{
	compute_direction(handle->context->get_stream(), handle->context->get_handle(), batch_size,
		batched_previous_location, *batched_previous_location_rows, *batched_previous_location_cols, *batched_previous_location_stride,
		batched_current_location, *batched_current_location_rows, *batched_current_location_cols, *batched_current_location_stride,
		batched_direction, *batched_direction_rows, *batched_direction_cols, *batched_direction_stride
	);

	compute_reachable_locations(handle->context->get_stream(), handle->context->get_handle(),
		batch_size, place_cells_number,
		rows_begin, rows_end,
		cols_begin, cols_end,
		radius, cos_half_angle,scale,seed,
		x_grid, x_grid_rows, x_grid_cols, x_grid_stride,
		y_grid, y_grid_rows, y_grid_cols, y_grid_stride,
		batched_current_location, *batched_current_location_rows, *batched_current_location_cols, *batched_current_location_stride,
		(const float **)batched_direction, *batched_direction_rows,*batched_direction_cols, *batched_direction_stride,
		batched_x_grid_centered, *batched_x_grid_centered_rows, *batched_x_grid_centered_cols, *batched_x_grid_centered_stride,
		batched_y_grid_centered, *batched_y_grid_centered_rows, *batched_y_grid_centered_cols, *batched_y_grid_centered_stride,
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
	const std::size_t *roi_row_begin, const std::size_t *roi_row_end, const std::size_t *roi_col_begin, const std::size_t *roi_col_end,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides,
	float **batched_predicted_location, const std::size_t *batched_predicted_location_rows, const std::size_t *batched_predicted_location_cols, const std::size_t *batched_predicted_location_strides

)
{
	/*compute_select_most_probable_location(handle->context->get_stream(), handle->context->get_handle(), batch_size, rows, cols,
		x_grid, x_grid_rows, x_grid_cols, x_grid_stride,
		y_grid, y_grid_rows, y_grid_cols, y_grid_stride,
		batched_location_probability, *batched_location_probability_rows, *batched_location_probability_cols, *batched_location_probability_strides,
		batched_predicted_location, *batched_predicted_location_rows, *batched_predicted_location_cols, *batched_predicted_location_strides
	);*/
}
void TRN::GPU::Algorithm::learn_widrow_hoff(
	const std::size_t &batch_size, const std::size_t &mini_batch_size,
	unsigned long &seed,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_w_ffwd, const std::size_t *batched_w_ffwd_rows, const std::size_t *batched_w_ffwd_cols, const std::size_t *batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const std::size_t *batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_w_rec, const std::size_t *batched_w_rec_rows, const std::size_t *batched_w_rec_cols, const std::size_t *batched_w_rec_strides,
	float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const std::size_t *batched_u_strides,
	float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const std::size_t *batched_p_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	float **batched_pre, const std::size_t *batched_pre_rows, const std::size_t *batched_pre_cols, const std::size_t *batched_pre_strides,
	float **batched_post, const std::size_t *batched_post_rows, const std::size_t *batched_post_cols, const std::size_t *batched_post_strides,
	float **batched_desired, const std::size_t *batched_desired_rows, const std::size_t *batched_desired_cols, const std::size_t *batched_desired_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
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
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_w_rec, *batched_w_rec_rows, *batched_w_rec_cols, *batched_w_rec_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p,* batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_pre, *batched_pre_rows, *batched_pre_cols, *batched_pre_strides,
			batched_post, *batched_post_rows, *batched_post_cols, *batched_post_strides,
			offsets, durations, repetitions, total_duration,
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
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_w_rec, *batched_w_rec_rows, *batched_w_rec_cols, *batched_w_rec_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p, *batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_pre, *batched_pre_rows, *batched_pre_cols, *batched_pre_strides,
			batched_post, *batched_post_rows, *batched_post_cols, *batched_post_strides,
			offsets, durations, repetitions, total_duration,
			states_samples, states_rows, states_cols, states_stride);
	}
}

void TRN::GPU::Algorithm::prime(
	const std::size_t &batch_size, const std::size_t &mini_batch_size,
	unsigned long &seed,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_w_ffwd, const std::size_t *batched_w_ffwd_rows, const std::size_t *batched_w_ffwd_cols, const std::size_t *batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const std::size_t *batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_w_rec, const std::size_t *batched_w_rec_rows, const std::size_t *batched_w_rec_cols, const std::size_t *batched_w_rec_strides,
	float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const std::size_t *batched_u_strides,
	float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const std::size_t *batched_p_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	float **batched_pre, const std::size_t *batched_pre_rows, const std::size_t *batched_pre_cols, const std::size_t *batched_pre_strides,
	float **batched_post, const std::size_t *batched_post_rows, const std::size_t *batched_post_cols, const std::size_t *batched_post_strides,
	float **batched_desired, const std::size_t *batched_desired_rows, const std::size_t *batched_desired_cols, const std::size_t *batched_desired_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions,const std::size_t &total_duration,
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
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_w_rec, *batched_w_rec_rows, *batched_w_rec_cols, *batched_w_rec_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p, *batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_pre, *batched_pre_rows, *batched_pre_cols, *batched_pre_strides,
			batched_post, *batched_post_rows, *batched_post_cols, *batched_post_strides,
			offsets, durations, repetitions, total_duration,
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
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_w_rec, *batched_w_rec_rows, *batched_w_rec_cols, *batched_w_rec_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p, *batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_pre, *batched_pre_rows, *batched_pre_cols, *batched_pre_strides,
			batched_post, *batched_post_rows, *batched_post_cols, *batched_post_strides,
			offsets, durations, repetitions, total_duration,
			states_samples, states_rows, states_cols, states_stride);
	}
		
}

void TRN::GPU::Algorithm::generate(
	const std::size_t &batch_size, const std::size_t &mini_batch_size,
	unsigned long &seed,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_w_ffwd, const std::size_t *batched_w_ffwd_rows, const std::size_t *batched_w_ffwd_cols, const std::size_t *batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const std::size_t *batched_u_ffwd_strides,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_w_rec, const std::size_t *batched_w_rec_rows, const std::size_t *batched_w_rec_cols, const std::size_t *batched_w_rec_strides,
	float **batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const std::size_t *batched_u_strides,
	float **batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const std::size_t *batched_p_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides,
	float **batched_pre, const std::size_t *batched_pre_rows, const std::size_t *batched_pre_cols, const std::size_t *batched_pre_strides,
	float **batched_post, const std::size_t *batched_post_rows, const std::size_t *batched_post_cols, const std::size_t *batched_post_strides,
	float **batched_desired, const std::size_t *batched_desired_rows, const std::size_t *batched_desired_cols, const std::size_t *batched_desired_strides,
	const int *offsets, const int *durations, const std::size_t &repetitions, const std::size_t &total_duration,
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
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_w_rec, *batched_w_rec_rows, *batched_w_rec_cols, *batched_w_rec_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p, *batched_p_rows, *batched_p_cols, *batched_p_strides,
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_pre, *batched_pre_rows, *batched_pre_cols, *batched_pre_strides,
			batched_post, *batched_post_rows, *batched_post_cols, *batched_post_strides,
			offsets, durations, repetitions,total_duration,
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
			batched_x_res, *batched_x_res_rows, *batched_x_res_cols, *batched_x_res_strides,
			batched_w_rec, *batched_w_rec_rows, *batched_w_rec_cols, *batched_w_rec_strides,
			batched_u, *batched_u_rows, *batched_u_cols, *batched_u_strides,
			batched_p, *batched_p_rows, *batched_p_cols, *batched_p_strides,
	
			batched_x_ro, *batched_x_ro_rows, *batched_x_ro_cols, *batched_x_ro_strides,
			batched_w_ro, *batched_w_ro_rows, *batched_w_ro_cols, *batched_w_ro_strides,
			batched_pre, *batched_pre_rows, *batched_pre_cols, *batched_pre_strides,
			batched_post, *batched_post_rows, *batched_post_cols, *batched_post_strides,
			offsets, durations, repetitions, total_duration,
			states_samples, states_rows, states_cols, states_stride);
	}
}

std::shared_ptr<TRN::GPU::Algorithm> TRN::GPU::Algorithm::create(const std::shared_ptr<TRN::GPU::Context> context)
{
	return std::make_shared<TRN::GPU::Algorithm>(context);
}
