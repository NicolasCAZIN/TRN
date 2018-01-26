#pragma once

#include "gpu_global.h"

#include "Backend/Algorithm.h"
#include "Context.h"

namespace TRN
{
	namespace GPU
	{
		class GPU_EXPORT Algorithm : public TRN::Backend::Algorithm
		{
		private:
			class Handle;
			mutable std::unique_ptr<Handle> handle;

		public:
			Algorithm(const std::shared_ptr<Context> context);
			~Algorithm();

		public:

			virtual void preallocate(const std::size_t &stimulus_size, const std::size_t &reservoir_size,
				const std::size_t &prediction_size, const std::size_t &batch_size) override;

			virtual void mean_square_error
			(
				const std::size_t &batch_size,
				const float **batched_predicted, const std::size_t *batched_predicted_rows, const std::size_t *batched_predicted_cols, const std::size_t *batched_predicted_strides,
				const float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
				float *result, const std::size_t &result_rows, const std::size_t &result_cols, const std::size_t &result_stride
			) override;



			virtual void place_cell_location_probability(
				const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
				const float &sigma,
				const float ** firing_rate_map, const std::size_t *firing_rate_map_rows, const std::size_t *firing_rate_map_cols, const std::size_t *firing_rate_map_strides,
				float **scale, const std::size_t *scale_rows, const std::size_t *scale_cols, const std::size_t *scale_strides,
				const float **prediction, const std::size_t *prediction_rows, const std::size_t *prediction_cols, const std::size_t *prediction_strides,
				float *** hypothesis_map, const std::size_t **hypothesis_map_rows, const std::size_t **hypothesis_map_cols, const std::size_t **hypothesis_map_strides,
				float ** location_probability, const std::size_t *location_probability_rows, const std::size_t *location_probability_cols, const std::size_t *location_probability_strides) override;

			virtual void restrict_to_reachable_locations(

				const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
				const float &radius, const float &cos_half_angle, const float &scale, const unsigned long &seed,
				const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
				const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
				const float **batched_previous_location, const std::size_t *batched_previous_location_rows, const std::size_t *batched_previous_location_cols, const std::size_t *batched_previous_location_stride,
				const float **batched_current_location, const std::size_t *batched_current_location_rows, const std::size_t *batched_current_location_cols, const std::size_t *batched_current_location_stride,
				float **batched_direction, const std::size_t *batched_direction_rows, const std::size_t *batched_direction_cols, const std::size_t *batched_direction_stride,
				float **batched_x_grid_centered, const std::size_t *batched_x_grid_centered_rows, const std::size_t *batched_x_grid_centered_cols, const std::size_t *batched_x_grid_centered_stride,
				float **batched_y_grid_centered, const std::size_t *batched_y_grid_centered_rows, const std::size_t *batched_y_grid_centered_cols, const std::size_t *batched_y_grid_centered_stride,
				float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides) override;


			virtual void draw_probable_location(const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
				const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
				const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
				const float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides,
				float **batched_reduced_location_probability, const std::size_t *batched_reduced_location_probability_rows, const std::size_t *batched_reduced_location_probability_cols, const std::size_t *batched_reduced_location_probability_stride,
				float **batched_row_cumsum, const std::size_t *batched_row_cumsum_rows, const std::size_t *batched_row_cumsum_cols, const std::size_t *batched_reduced_row_cumsum_strides,
				float **batched_col_cumsum, const std::size_t *batched_col_cumsum_rows, const std::size_t *batched_col_cumsum_cols, const std::size_t *batched_reduced_col_cumsum_strides,
				float **batched_predicted_location, const std::size_t *batched_predicted_location_rows, const std::size_t *batched_predicted_location_cols, const std::size_t *batched_predicted_location_strides
			) override;



			virtual void select_most_probable_location(const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
				const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
				const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
				const float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides,
				float **batched_predicted_location, const std::size_t *batched_predicted_location_rows, const std::size_t *batched_predicted_location_cols, const std::size_t *batched_predicted_location_strides
			) override;

			virtual void learn_widrow_hoff(
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
				float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_stride,
				const int *offsets, const int *durations, const std::size_t &repetitions,
				float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,

				const float &learning_rate
			) override;
			virtual void prime(
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
				float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_stride,
				const int *offsets, const int *durations, const std::size_t &repetitions,
				float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride) override;
			virtual void generate(
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
				float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_stride,
				const int *offsets, const int *durations, const std::size_t &repetitions,
				float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride) override;

		public:
			static std::shared_ptr<Algorithm> create(const std::shared_ptr<Context> context);
		};
	};
};
