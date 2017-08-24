#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Widrow_Hoff
{
private :
	const float learning_rate;
public:
	CUDA_CALLABLE_MEMBER Widrow_Hoff(const float &learning_rate) : learning_rate(learning_rate) {}
	CUDA_CALLABLE_MEMBER const float &get_learning_rate() const
	{
		return learning_rate;
	}
};
class Nothing
{
};

 void compute_mean_square_error(
	 const cudaStream_t &stream,
	 const cublasHandle_t &handle,
	 const std::size_t &batch_size,
	 const float **batched_predicted, const std::size_t &batched_predicted_rows, const std::size_t &batched_predicted_cols, const std::size_t &batched_predicted_stride,
	 const float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	 float *result, const std::size_t &result_rows, const std::size_t &result_cols, const std::size_t &result_stride);

void compute_place_cell_location_probability(
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float &sigma,
	const float ** firing_rate_map, const std::size_t &firing_rate_map_rows, const std::size_t &firing_rate_map_cols, const std::size_t &firing_rate_map_stride,
	float **scale, const std::size_t &scale_rows, const std::size_t &scale_cols, const std::size_t &scale_stride,
	const float **prediction, const std::size_t &prediction_rows, const std::size_t &prediction_cols, const std::size_t &prediction_stride,
	float *** hypothesis_map, const std::size_t &hypothesis_map_rows, const std::size_t &hypothesis_map_cols, const std::size_t &hypothesis_map_stride,
	float ** location_probability, const std::size_t &location_probability_rows, const std::size_t &location_probability_cols, const std::size_t &location_probability_stride
);

void compute_reachable_locations(
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float &radius,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float **batched_current_location, const std::size_t &batched_current_location_rows, const std::size_t &batched_current_location_cols, const std::size_t &batched_current_location_stride,
	float **batched_x_grid_centered2, const std::size_t &batched_x_grid_centered2_rows, const std::size_t &batched_x_grid_centered2_cols, const std::size_t &batched_x_grid_centered2_stride,
	float **batched_y_grid_centered2, const std::size_t &batched_y_grid_centered2_rows, const std::size_t &batched_y_grid_centered2_cols, const std::size_t &batched_y_grid_centered2_stride,
	float  **batched_location_probability, const std::size_t &batched_location_probability_rows, const std::size_t &batched_location_probability_cols, const std::size_t &batched_location_probability_strides);

void compute_select_most_probable_location(const cudaStream_t &stream, const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t &batched_location_probability_rows, const std::size_t &batched_location_probability_cols, const std::size_t &batched_location_probability_strides,
	float **batched_predicted_location, const std::size_t &batched_predicted_location_rows, const std::size_t &batched_predicted_location_cols, const std::size_t &batched_predicted_location_strides
);

void compute_draw_probable_location(
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t &batched_location_probability_rows, const std::size_t &batched_location_probability_cols, const std::size_t &batched_location_probability_strides,
	float **batched_reduced_location_probability, const std::size_t &batched_reduced_location_probability_rows, const std::size_t &batched_reduced_location_probability_cols, const std::size_t &batched_reduced_location_probability_stride,
	float **batched_row_cumsum, const std::size_t &batched_row_cumsum_rows, const std::size_t &batched_row_cumsum_cols, const std::size_t &batched_row_cumsum_strides,
	float **batched_col_cumsum, const std::size_t &batched_col_cumsum_rows, const std::size_t &batched_col_cumsum_cols, const std::size_t &batched_col_cumsum_strides,
	float **batched_predicted_location, const std::size_t &batched_predicted_location_rows, const std::size_t &batched_predicted_location_cols, const std::size_t &batched_predicted_location_strides);

template<bool gather_states, bool overwrite_states, typename Parameter>
void update_model(
	const std::size_t &batch_size,
	unsigned long &seed,
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const Parameter &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	float **batched_w_ffwd, const std::size_t &batched_w_ffwd_rows, const std::size_t &batched_w_ffwd_cols, const std::size_t &batched_w_ffwd_strides,
	float **batched_u_ffwd, const std::size_t &batched_u_ffwd_rows, const std::size_t &batched_u_ffwd_cols, const std::size_t &batched_u_ffwd_strides,
	float **batched_x_in, const std::size_t &batched_x_in_rows, const std::size_t &batched_x_in_cols, const std::size_t &batched_x_in_strides,
	float **batched_w_in, const std::size_t &batched_w_in_rows, const std::size_t &batched_w_in_cols, const std::size_t &batched_w_in_strides,
	float **batched_u, const std::size_t &batched_u_rows, const std::size_t &batched_u_cols, const std::size_t &batched_u_strides,
	float **batched_p, const std::size_t &batched_p_rows, const std::size_t &batched_p_cols, const std::size_t &batched_p_strides,
	float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t &batched_w_ro_strides,
	float **batched_error, const std::size_t & batched_error_rows, const std::size_t &batched_error_cols, const std::size_t &batched_error_strides,
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride);