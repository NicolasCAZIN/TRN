#pragma once

#include "Implementation.h"
#include "Algorithm.h"
#include "Memory.h"
#include "Random.h"
#include <ctime>
#include <vector>



class Widrow_Hoff
{
private:
	const float learning_rate;
public:
	 Widrow_Hoff(const float &learning_rate) : learning_rate(learning_rate) {}
	 const float &get_learning_rate() const
	{
		return learning_rate;
	}
};
class Nothing
{
};

template <TRN::CPU::Implementation Implementation>
TRN::CPU::Algorithm<Implementation>::Algorithm() 
{

}

template <TRN::CPU::Implementation Implementation>
TRN::CPU::Algorithm<Implementation>::~Algorithm()
{

}
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::preallocate(const std::size_t &stimulus_size, const std::size_t &reservoir_size,
	const std::size_t &prediction_size, const std::size_t &batch_size)
{

}


template <TRN::CPU::Implementation Implementation>
static inline float 	compute_mse(const float *A, const float *B, const size_t &cols)
{
	std::size_t col = 0;
	typename TRN::CPU::Traits<Implementation>::type acc = setzero_ps();
	if (cols - col > _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			auto _d0 = sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]));
			acc = mul_add_ps(_d0, _d0, acc);
			auto _d1 = sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]));
			acc = mul_add_ps(_d1, _d1, acc);
			auto _d2 = sub_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2]));
			acc = mul_add_ps(_d2, _d2, acc);
			auto _d3 = sub_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3]));
			acc = mul_add_ps(_d3, _d3, acc);
			auto _d4 = sub_ps(load_ps(&A[col + _4]), load_ps(&B[col + _4]));
			acc = mul_add_ps(_d4, _d4, acc);
			auto _d5 = sub_ps(load_ps(&A[col + _5]), load_ps(&B[col + _5]));
			acc = mul_add_ps(_d5, _d5, acc);
			auto _d6 = sub_ps(load_ps(&A[col + _6]), load_ps(&B[col + _6]));
			acc = mul_add_ps(_d6, _d6, acc);
			auto _d7 = sub_ps(load_ps(&A[col + _7]), load_ps(&B[col + _7]));
			acc = mul_add_ps(_d7, _d7, acc);
		}
	}
	if (cols - col > _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			auto _d0 = sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]));
			acc = mul_add_ps(_d0, _d0, acc);
			auto _d1 = sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]));
			acc = mul_add_ps(_d1, _d1, acc);
			auto _d2 = sub_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2]));
			acc = mul_add_ps(_d2, _d2, acc);
			auto _d3 = sub_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3]));
			acc = mul_add_ps(_d3, _d3, acc);
		}
	}
	if (cols - col > _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			auto _d0 = sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]));
			acc = mul_add_ps(_d0, _d0, acc);
			auto _d1 = sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]));
			acc = mul_add_ps(_d1, _d1, acc);
		}
	}
	if (cols - col > _1)
	{
		for (; col + _1 - 1 < cols; col += _1)
		{
			auto _d0 = sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]));
			acc = mul_add_ps(_d0, _d0, acc);
		}
	}

	auto e = hsum_ps(acc);

	if (cols - col > 0)
	{
		for (; col  < cols; col++)
		{
			auto d = (A[col] - B[col]);
			e += d * d;
		}
	}


	return e / cols;
}

template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::mean_square_error
(
	const std::size_t &batch_size,
	const float **batched_predicted, const std::size_t *batched_predicted_rows, const std::size_t *batched_predicted_cols, const std::size_t *batched_predicted_strides,
	const float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float *result, const std::size_t &result_rows, const std::size_t &result_cols, const std::size_t &result_stride
)
{
	int K = expected_rows * batch_size;
#pragma omp parallel for
	for (int k = 0; k < K; k++)
	{
		int batch = k % batch_size;
		int row = k / batch_size;
		const float *predicted = batched_predicted[batch];
		const std::size_t predicted_rows = batched_predicted_rows[batch];
		const std::size_t predicted_cols = batched_predicted_cols[batch];
		const std::size_t predicted_stride = batched_predicted_strides[batch];		
		assert(predicted_cols == expected_cols);
		assert(predicted_rows == expected_rows);
		auto mse = compute_mse<Implementation>(&predicted[row * predicted_stride], &expected[row * expected_stride], expected_cols);

		result[row * result_stride + batch] = mse;
	}
}

template <TRN::CPU::Implementation Implementation>
static inline void 	weighted_sum(
	const float *A, const typename TRN::CPU::Traits<Implementation>::type  &a,
	const float *B, const typename TRN::CPU::Traits<Implementation>::type  &b,
	const std::size_t &cols, float *C)
{
	std::size_t col = 0;
	
	if (cols - col > _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, mul_ps(load_ps(&B[col + _1]), b)));
			stream_ps(&C[col + _2], mul_add_ps(load_ps(&A[col + _2]), a, mul_ps(load_ps(&B[col + _2]), b)));
			stream_ps(&C[col + _3], mul_add_ps(load_ps(&A[col + _3]), a, mul_ps(load_ps(&B[col + _3]), b)));
			stream_ps(&C[col + _4], mul_add_ps(load_ps(&A[col + _4]), a, mul_ps(load_ps(&B[col + _4]), b)));
			stream_ps(&C[col + _5], mul_add_ps(load_ps(&A[col + _5]), a, mul_ps(load_ps(&B[col + _5]), b)));
			stream_ps(&C[col + _6], mul_add_ps(load_ps(&A[col + _6]), a, mul_ps(load_ps(&B[col + _6]), b)));
			stream_ps(&C[col + _7], mul_add_ps(load_ps(&A[col + _7]), a, mul_ps(load_ps(&B[col + _7]), b)));

		}
	}
	if (cols - col > _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, mul_ps(load_ps(&B[col + _1]), b)));
			stream_ps(&C[col + _2], mul_add_ps(load_ps(&A[col + _2]), a, mul_ps(load_ps(&B[col + _2]), b)));
			stream_ps(&C[col + _3], mul_add_ps(load_ps(&A[col + _3]), a, mul_ps(load_ps(&B[col + _3]), b)));
		}
	}
	if (cols - col > _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, mul_ps(load_ps(&B[col + _1]), b)));
		}
	}
	if (cols - col > 0)
	{
		for (; col < cols; col += _1)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
		}
	}

	/*if (cols - col > _1)
	{
		for (; col + _1 - 1 < cols; col += _1)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
		}
	}
	if (cols - col > 0)
	{
		float _a = get_element(0, a);
		float _b = get_element(0, b);
		for (; col  < cols; col ++)
		{
			C[col] = A[col] * _a + B[col] * _b;
			//stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
		}
	}*/
}
template <TRN::CPU::Implementation Implementation>
static inline void 	sub_scale(
	const float *A,
	const float *B,
	const typename TRN::CPU::Traits<Implementation>::type &scale,
	const std::size_t &cols, float *C)
{
	std::size_t col = 0;
	/*if (cols - col > 06)
	{
		for (; col + _16 - 1 < cols; col += _16)
		{
			stream_ps(&C[col + _0], mul_ps(scale, sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]))));
			stream_ps(&C[col + _1], mul_ps(scale, sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]))));
			stream_ps(&C[col + _2], mul_ps(scale, sub_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2]))));
			stream_ps(&C[col + _3], mul_ps(scale, sub_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3]))));
			stream_ps(&C[col + _4], mul_ps(scale, sub_ps(load_ps(&A[col + _4]), load_ps(&B[col + _4]))));
			stream_ps(&C[col + _5], mul_ps(scale, sub_ps(load_ps(&A[col + _5]), load_ps(&B[col + _5]))));
			stream_ps(&C[col + _6], mul_ps(scale, sub_ps(load_ps(&A[col + _6]), load_ps(&B[col + _6]))));
			stream_ps(&C[col + _7], mul_ps(scale, sub_ps(load_ps(&A[col + _7]), load_ps(&B[col + _7]))));
			stream_ps(&C[col + _8], mul_ps(scale, sub_ps(load_ps(&A[col + _8]), load_ps(&B[col + _8]))));
			stream_ps(&C[col + _9], mul_ps(scale, sub_ps(load_ps(&A[col + _9]), load_ps(&B[col + _9]))));
			stream_ps(&C[col + _10], mul_ps(scale, sub_ps(load_ps(&A[col + _10]), load_ps(&B[col + _10]))));
			stream_ps(&C[col + _11], mul_ps(scale, sub_ps(load_ps(&A[col + _11]), load_ps(&B[col + _11]))));
			stream_ps(&C[col + _12], mul_ps(scale, sub_ps(load_ps(&A[col + _12]), load_ps(&B[col + _12]))));
			stream_ps(&C[col + _13], mul_ps(scale, sub_ps(load_ps(&A[col + _13]), load_ps(&B[col + _13]))));
			stream_ps(&C[col + _14], mul_ps(scale, sub_ps(load_ps(&A[col + _14]), load_ps(&B[col + _14]))));
			stream_ps(&C[col + _15], mul_ps(scale, sub_ps(load_ps(&A[col + _15]), load_ps(&B[col + _15]))));
		}
	}*/
	if (cols - col > _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&C[col + _0], mul_ps(scale, sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]))));
			stream_ps(&C[col + _1], mul_ps(scale, sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]))));
			stream_ps(&C[col + _2], mul_ps(scale, sub_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2]))));
			stream_ps(&C[col + _3], mul_ps(scale, sub_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3]))));
			stream_ps(&C[col + _4], mul_ps(scale, sub_ps(load_ps(&A[col + _4]), load_ps(&B[col + _4]))));
			stream_ps(&C[col + _5], mul_ps(scale, sub_ps(load_ps(&A[col + _5]), load_ps(&B[col + _5]))));
			stream_ps(&C[col + _6], mul_ps(scale, sub_ps(load_ps(&A[col + _6]), load_ps(&B[col + _6]))));
			stream_ps(&C[col + _7], mul_ps(scale, sub_ps(load_ps(&A[col + _7]), load_ps(&B[col + _7]))));
		}
	}
	if (cols - col > _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&C[col + _0], mul_ps(scale, sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]))));
			stream_ps(&C[col + _1], mul_ps(scale, sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]))));
			stream_ps(&C[col + _2], mul_ps(scale, sub_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2]))));
			stream_ps(&C[col + _3], mul_ps(scale, sub_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3]))));
		}
	}
	if (cols - col > _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&C[col + _0], mul_ps(scale, sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]))));
			stream_ps(&C[col + _1], mul_ps(scale, sub_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1]))));
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			stream_ps(&C[col + _0], mul_ps(scale, sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]))));
		}
	}

}
template <TRN::CPU::Implementation Implementation>
static inline void 	sum(
	const float *A,
	const float *B,
	const std::size_t &cols, float *C)
{
	std::size_t col = 0;
	/*if (cols - col > 06)
	{
		for (; col + _16 - 1 < cols; col += _16)
		{
			stream_ps(&C[col + _0], add_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0])));
			stream_ps(&C[col + _1], add_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1])));
			stream_ps(&C[col + _2], add_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2])));
			stream_ps(&C[col + _3], add_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3])));
			stream_ps(&C[col + _4], add_ps(load_ps(&A[col + _4]), load_ps(&B[col + _4])));
			stream_ps(&C[col + _5], add_ps(load_ps(&A[col + _5]), load_ps(&B[col + _5])));
			stream_ps(&C[col + _6], add_ps(load_ps(&A[col + _6]), load_ps(&B[col + _6])));
			stream_ps(&C[col + _7], add_ps(load_ps(&A[col + _7]), load_ps(&B[col + _7])));
			stream_ps(&C[col + _8], add_ps(load_ps(&A[col + _8]), load_ps(&B[col + _8])));
			stream_ps(&C[col + _9], add_ps(load_ps(&A[col + _9]), load_ps(&B[col + _9])));
			stream_ps(&C[col + _10], add_ps(load_ps(&A[col + _10]), load_ps(&B[col + _10])));
			stream_ps(&C[col + _11], add_ps(load_ps(&A[col + _11]), load_ps(&B[col + _11])));
			stream_ps(&C[col + _12], add_ps(load_ps(&A[col + _12]), load_ps(&B[col + _12])));
			stream_ps(&C[col + _13], add_ps(load_ps(&A[col + _13]), load_ps(&B[col + _13])));
			stream_ps(&C[col + _14], add_ps(load_ps(&A[col + _14]), load_ps(&B[col + _14])));
			stream_ps(&C[col + _15], add_ps(load_ps(&A[col + _15]), load_ps(&B[col + _15])));
		}
	}*/
	if (cols - col > _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&C[col + _0], add_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0])));
			stream_ps(&C[col + _1], add_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1])));
			stream_ps(&C[col + _2], add_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2])));
			stream_ps(&C[col + _3], add_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3])));
			stream_ps(&C[col + _4], add_ps(load_ps(&A[col + _4]), load_ps(&B[col + _4])));
			stream_ps(&C[col + _5], add_ps(load_ps(&A[col + _5]), load_ps(&B[col + _5])));
			stream_ps(&C[col + _6], add_ps(load_ps(&A[col + _6]), load_ps(&B[col + _6])));
			stream_ps(&C[col + _7], add_ps(load_ps(&A[col + _7]), load_ps(&B[col + _7])));
		}
	}
	if (cols - col > _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&C[col + _0], add_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0])));
			stream_ps(&C[col + _1], add_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1])));
			stream_ps(&C[col + _2], add_ps(load_ps(&A[col + _2]), load_ps(&B[col + _2])));
			stream_ps(&C[col + _3], add_ps(load_ps(&A[col + _3]), load_ps(&B[col + _3])));
		}
	}
	if (cols - col > _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&C[col + _0], add_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0])));
			stream_ps(&C[col + _1], add_ps(load_ps(&A[col + _1]), load_ps(&B[col + _1])));
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			stream_ps(&C[col + _0], add_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0])));
		}
	}

}

template <TRN::CPU::Implementation Implementation>
static inline void 	weighted_acc(const float *A, const typename TRN::CPU::Traits<Implementation>::type  &a, const std::size_t &cols, float *C)
{
	std::size_t col = 0;
	
	if (cols - col > _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, load_ps(&C[col + _0])));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, load_ps(&C[col + _1])));
			stream_ps(&C[col + _2], mul_add_ps(load_ps(&A[col + _2]), a, load_ps(&C[col + _2])));
			stream_ps(&C[col + _3], mul_add_ps(load_ps(&A[col + _3]), a, load_ps(&C[col + _3])));
			stream_ps(&C[col + _4], mul_add_ps(load_ps(&A[col + _4]), a, load_ps(&C[col + _4])));
			stream_ps(&C[col + _5], mul_add_ps(load_ps(&A[col + _5]), a, load_ps(&C[col + _5])));
			stream_ps(&C[col + _6], mul_add_ps(load_ps(&A[col + _6]), a, load_ps(&C[col + _6])));
			stream_ps(&C[col + _7], mul_add_ps(load_ps(&A[col + _7]), a, load_ps(&C[col + _7])));
		}
	}
	if (cols - col > _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, load_ps(&C[col + _0])));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, load_ps(&C[col + _1])));
			stream_ps(&C[col + _2], mul_add_ps(load_ps(&A[col + _2]), a, load_ps(&C[col + _2])));
			stream_ps(&C[col + _3], mul_add_ps(load_ps(&A[col + _3]), a, load_ps(&C[col + _3])));
		}
	}
	if (cols - col > _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, load_ps(&C[col + _0])));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, load_ps(&C[col + _1])));
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, load_ps(&C[col + _0])));
		}
	}
}


template <TRN::CPU::Implementation Implementation>
static inline void 	diff_square(const typename TRN::CPU::Traits<Implementation>::type &current, const float *grid, const std::size_t &cols, float *grid_centered2)
{
	std::size_t col = 0;
	
	if (cols - col > _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&grid_centered2[col + _0], sqr_ps(sub_ps(load_ps(&grid[col + _0]), current)));
			stream_ps(&grid_centered2[col + _1], sqr_ps(sub_ps(load_ps(&grid[col + _1]), current)));
			stream_ps(&grid_centered2[col + _2], sqr_ps(sub_ps(load_ps(&grid[col + _2]), current)));
			stream_ps(&grid_centered2[col + _3], sqr_ps(sub_ps(load_ps(&grid[col + _3]), current)));
			stream_ps(&grid_centered2[col + _4], sqr_ps(sub_ps(load_ps(&grid[col + _4]), current)));
			stream_ps(&grid_centered2[col + _5], sqr_ps(sub_ps(load_ps(&grid[col + _5]), current)));
			stream_ps(&grid_centered2[col + _6], sqr_ps(sub_ps(load_ps(&grid[col + _6]), current)));
			stream_ps(&grid_centered2[col + _7], sqr_ps(sub_ps(load_ps(&grid[col + _7]), current)));
		}
	}
	if (cols - col > _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&grid_centered2[col + _0], sqr_ps(sub_ps(load_ps(&grid[col + _0]), current)));
			stream_ps(&grid_centered2[col + _1], sqr_ps(sub_ps(load_ps(&grid[col + _1]), current)));
			stream_ps(&grid_centered2[col + _2], sqr_ps(sub_ps(load_ps(&grid[col + _2]), current)));
			stream_ps(&grid_centered2[col + _3], sqr_ps(sub_ps(load_ps(&grid[col + _3]), current)));
		}
	}
	if (cols - col > _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&grid_centered2[col + _0], sqr_ps(sub_ps(load_ps(&grid[col + _0]), current)));
			stream_ps(&grid_centered2[col + _1], sqr_ps(sub_ps(load_ps(&grid[col + _1]), current)));
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			stream_ps(&grid_centered2[col + _0], sqr_ps(sub_ps(load_ps(&grid[col + _0]), current)));
		}
	}

}

template <TRN::CPU::Implementation Implementation>
static inline void 	circle(VSLStreamStatePtr &stream, const float &scale, float *x_grid_centered2, const std::size_t &cols, const typename TRN::CPU::Traits<Implementation>::type &y2, const typename TRN::CPU::Traits<Implementation>::type &r2, float *location_probability_row)
{
	//	 vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, 1000, r, 0.0, scale);
	const auto __zero = setzero_ps();
	std::size_t col = 0;

	if (cols - col > _8)
	{
		float noise[_8];
		for (; col + _8 - 1 < cols; col += _8)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _8, noise, 0.0, scale);
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _0]), y2), r2)));
			stream_ps(&location_probability_row[col + _1], blendv_ps(__zero, add_ps(load_ps(&noise[_1]), load_ps(&location_probability_row[col + _1])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _1]), y2), r2)));
			stream_ps(&location_probability_row[col + _2], blendv_ps(__zero, add_ps(load_ps(&noise[_2]), load_ps(&location_probability_row[col + _2])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _2]), y2), r2)));
			stream_ps(&location_probability_row[col + _3], blendv_ps(__zero, add_ps(load_ps(&noise[_3]), load_ps(&location_probability_row[col + _3])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _3]), y2), r2)));
			stream_ps(&location_probability_row[col + _4], blendv_ps(__zero, add_ps(load_ps(&noise[_4]), load_ps(&location_probability_row[col + _4])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _4]), y2), r2)));
			stream_ps(&location_probability_row[col + _5], blendv_ps(__zero, add_ps(load_ps(&noise[_5]), load_ps(&location_probability_row[col + _5])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _5]), y2), r2)));
			stream_ps(&location_probability_row[col + _6], blendv_ps(__zero, add_ps(load_ps(&noise[_6]), load_ps(&location_probability_row[col + _6])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _6]), y2), r2)));
			stream_ps(&location_probability_row[col + _7], blendv_ps(__zero, add_ps(load_ps(&noise[_7]), load_ps(&location_probability_row[col + _7])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _7]), y2), r2)));
		}
	}
	if (cols - col > _4)
	{
		float noise[_4];
		for (; col + _4 - 1 < cols; col += _4)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _4, noise, 0.0, scale);
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _0]), y2), r2)));
			stream_ps(&location_probability_row[col + _1], blendv_ps(__zero, add_ps(load_ps(&noise[_1]), load_ps(&location_probability_row[col + _1])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _1]), y2), r2)));
			stream_ps(&location_probability_row[col + _2], blendv_ps(__zero, add_ps(load_ps(&noise[_2]), load_ps(&location_probability_row[col + _2])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _2]), y2), r2)));
			stream_ps(&location_probability_row[col + _3], blendv_ps(__zero, add_ps(load_ps(&noise[_3]), load_ps(&location_probability_row[col + _3])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _3]), y2), r2)));
		}
	}
	if (cols - col > _2)
	{
		float noise[_2];
		for (; col + _2 - 1 < cols; col += _2)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _2, noise, 0.0, scale);
			stream_ps(&location_probability_row[col + _0],  blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _0]), y2), r2)));
			stream_ps(&location_probability_row[col + _1],  blendv_ps(__zero, add_ps(load_ps(&noise[_1]), load_ps(&location_probability_row[col + _1])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _1]), y2), r2)));
		}
	}
	 if (cols - col > 0)
	{
		 float noise[_1];
		for (; col  < cols; col += _1)
		{
			vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, _1, noise, 0.0, scale);
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, add_ps(load_ps(&noise[_0]), load_ps(&location_probability_row[col + _0])), cmp_lt_ps(add_ps(load_ps(&x_grid_centered2[col + _0]), y2), r2)));
		}
	}
}

template <TRN::CPU::Implementation Implementation>
static inline void location_hypothesis(const float *firing_rate_row, const std::size_t &cols,
	const typename TRN::CPU::Traits<Implementation>::type  &__prediction,
	const typename TRN::CPU::Traits<Implementation>::type  &__inv_sigma2,
	float *hypothesis_row)
{
	std::size_t col = 0;
	if (cols - col > _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			auto __h0 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _0], __h0);
			auto __h1 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _1]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _1], __h1);
			auto __h2 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _2]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _2], __h2);
			auto __h3 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _3]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _3], __h3);			
			auto __h4 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _4]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _4], __h4);
			auto __h5 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _5]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _5], __h5);
			auto __h6 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _6]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _6], __h6);
			auto __h7 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _7]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _7], __h7);
		}
	}
	if (cols - col > _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			auto __h0 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _0], __h0);
			auto __h1 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _1]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _1], __h1);
			auto __h2 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _2]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _2], __h2);
			auto __h3 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _3]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _3], __h3);
		}
	}
	if (cols - col > _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			auto __h0 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _0], __h0);
			auto __h1 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _1]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _1], __h1);
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			auto __h0 = mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2);
			stream_ps(&hypothesis_row[col + _0], __h0);
		}
	}
}

template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::place_cell_location_probability(
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float &sigma,
	const float ** firing_rate_map, const std::size_t *firing_rate_map_rows, const std::size_t *firing_rate_map_cols, const std::size_t *firing_rate_map_strides,
	float **scale, const std::size_t *scale_rows, const std::size_t *scale_cols, const std::size_t *scale_strides,
	const float **prediction, const std::size_t *prediction_rows, const std::size_t *prediction_cols, const std::size_t *prediction_strides,
	float *** hypothesis_map, const std::size_t **hypothesis_map_rows, const std::size_t **hypothesis_map_cols, const std::size_t **hypothesis_map_strides,
	float ** location_probability, const std::size_t *location_probability_rows, const std::size_t *location_probability_cols, const std::size_t *location_probability_strides)
{
	const float _inv_sigma2 = -1.0f / (sigma*sigma);
	const auto ___inv_sigma2 = set1_ps(_inv_sigma2);

	int K = batch_size * place_cells_number;
#pragma omp parallel for
	for (int k = 0; k < K; k++)
	{
		int batch = k % batch_size;
		int place_cell = k / batch_size;
		const std::size_t hypothesis_map_stride = hypothesis_map_strides[batch][place_cell];
		const std::size_t firing_rate_map_stride = firing_rate_map_strides[place_cell];
		auto firing_rate_k = firing_rate_map[place_cell];
		auto hypothesis_k = hypothesis_map[batch][place_cell];

		const float &p = prediction[batch][place_cell];
		const auto &__prediction = set1_ps(p);
		float sum = 0.0f;
		for (std::size_t row = 0; row < rows; row++)
		{
			auto firing_rate_row = &firing_rate_k[row * firing_rate_map_stride];
			auto hypothesis_row = &hypothesis_k[row * hypothesis_map_stride];

			location_hypothesis<Implementation>(firing_rate_row, hypothesis_map_stride, __prediction, ___inv_sigma2, hypothesis_row);
			vsExp(cols, hypothesis_row, hypothesis_row);
			sum += cblas_sasum(cols, hypothesis_row, 1);
		}
		if (sum > 0.0f)
			scale[batch][place_cell] = 1.0f / (sum * (float)place_cells_number);
		else
			scale[batch][place_cell] = 0.0f;
	}



	const std::size_t place_cells_number_range = place_cells_number / 2;
	const std::size_t place_cells_number_remaining = place_cells_number - place_cells_number_range * 2;

	K = place_cells_number_range * batch_size;

#pragma omp parallel for
	for (int k = 0; k < K; k++)
	{
		int batch = k % batch_size;
		int place_cell = k / batch_size;

		auto place_cell_a = place_cell;
		auto place_cell_b = place_cells_number_range + place_cell;
		auto hypothesis_a = hypothesis_map[batch][place_cell_a];
		auto hypothesis_b = hypothesis_map[batch][place_cell_b];
		auto stride_a = hypothesis_map_strides[batch][place_cell_a];
		auto stride_b = hypothesis_map_strides[batch][place_cell_b];
		const auto scale_a = set1_ps(scale[batch][place_cell_a]);
		const auto scale_b = set1_ps(scale[batch][place_cell_b]);
		for (std::size_t row = 0; row < rows; row++)
		{
			auto hypothesis_a_row = &hypothesis_a[row * stride_a];
			auto hypothesis_b_row = &hypothesis_b[row * stride_b];
			weighted_sum<Implementation>(hypothesis_a_row, scale_a, hypothesis_b_row, scale_b, cols, hypothesis_a_row);
		}
	}

	if (place_cells_number_range >= 2)
	{

		for (std::size_t range = place_cells_number_range / 2; range > 1; range /= 2)
		{
			K = range * batch_size;
#pragma omp parallel for
			for (int k = 0; k < K; k++)
			{
				int batch = k % batch_size;
				int place_cell = k / batch_size;
				auto place_cell_a = place_cell;
				auto place_cell_b = range + place_cell;
				auto hypothesis_a = hypothesis_map[batch][place_cell_a];
				auto hypothesis_b = hypothesis_map[batch][place_cell_b];
				auto stride_a = hypothesis_map_strides[batch][place_cell_a];
				auto stride_b = hypothesis_map_strides[batch][place_cell_b];
				for (std::size_t row = 0; row < rows; row++)
				{
					auto hypothesis_a_row = &hypothesis_a[row * stride_a];
					auto hypothesis_b_row = &hypothesis_b[row * stride_b];

					sum<Implementation>(hypothesis_a_row, hypothesis_b_row, cols, hypothesis_a_row);
				}
			}
		}
		{
			K = batch_size * rows;
#pragma omp parallel for
			for (int k = 0; k < K; k++)
			{
				int batch = k % batch_size;
				int row = k / batch_size;
				auto hypothesis_a = hypothesis_map[batch][0];
				auto hypothesis_b = hypothesis_map[batch][1];
				auto stride_a = hypothesis_map_strides[batch][0];
				auto stride_b = hypothesis_map_strides[batch][1];
				auto hypothesis_a_row = &hypothesis_a[row * stride_a];
				auto hypothesis_b_row = &hypothesis_b[row * stride_b];
				auto location_probability_row = &location_probability[batch][row * location_probability_strides[batch]];

				sum<Implementation>(hypothesis_a_row, hypothesis_b_row, cols, location_probability_row);
			}
		}
	}

	if (place_cells_number_remaining > 0)
	{
		K = batch_size * rows;
#pragma omp parallel for
		for (int k = 0; k < K; k++)
		{
			int batch = k % batch_size;
			int row = k / batch_size;
			auto hypothesis_k = hypothesis_map[batch][place_cells_number_range];
			auto scale_k = set1_ps(scale[batch][place_cells_number_range]);
			auto hypothesis_k_row = &hypothesis_k[row * hypothesis_map_strides[batch][place_cells_number_range]];
			auto location_probability_row = &location_probability[batch][row* location_probability_strides[batch]];

			weighted_acc<Implementation>(hypothesis_k_row, scale_k, cols, location_probability_row);
		}
	}
}

template <TRN::CPU::Implementation Implementation>
 void TRN::CPU::Algorithm<Implementation>::restrict_to_reachable_locations(
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float &radius, const float &scale, const unsigned long &seed,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float **batched_current_location, const std::size_t *batched_current_location_rows, const std::size_t *batched_current_location_cols, const std::size_t *batched_current_location_stride,
	float **batched_x_grid_centered2, const std::size_t *batched_x_grid_centered2_rows, const std::size_t *batched_x_grid_centered2_cols, const std::size_t *batched_x_grid_centered2_stride,
	float **batched_y_grid_centered2, const std::size_t *batched_y_grid_centered2_rows, const std::size_t *batched_y_grid_centered2_cols, const std::size_t *batched_y_grid_centered2_stride,
	float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides)
{

	 auto r2 = radius * radius;
#pragma omp parallel for
	 for (int batch = 0; batch < batch_size; batch++)
	 {

		 diff_square<Implementation>(set1_ps(batched_current_location[batch][0]), x_grid, x_grid_cols, batched_x_grid_centered2[batch]);
		 diff_square<Implementation>(set1_ps(batched_current_location[batch][1]), y_grid, y_grid_cols, batched_y_grid_centered2[batch]);
	 }
	 std::vector<VSLStreamStatePtr> streams(omp_get_max_threads());
#pragma omp parallel for
	 for (int tid = 0; tid < streams.size(); tid++)
	 {
		 vslNewStream(&streams[tid], VSL_BRNG_MT19937, seed);
	 }

	 



	 int K = batch_size * rows;
#pragma omp parallel for
	 for (int k = 0; k < K; k++)
	 {
		 auto tid = omp_get_thread_num();
		 int batch = k % batch_size;
		 int row = k / batch_size;

	
		 auto location_probability_row = &batched_location_probability[batch][row * batched_location_probability_strides[batch]];
		 auto y2 = batched_y_grid_centered2[batch][row];

		 circle<Implementation>(streams[tid], scale, batched_x_grid_centered2[batch], x_grid_cols, set1_ps(y2), set1_ps(r2), location_probability_row);

	 }

#pragma omp parallel for
	 for (int tid = 0; tid < streams.size(); tid++)
	 {
		 vslDeleteStream(&streams[tid]);
	 }

}

template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::draw_probable_location(const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides,
	float **batched_reduced_location_probability, const std::size_t *batched_reduced_location_probability_rows, const std::size_t *batched_reduced_location_probability_cols, const std::size_t *batched_reduced_location_probability_stride,
	float **batched_row_cumsum, const std::size_t *batched_row_cumsum_rows, const std::size_t *batched_row_cumsum_cols, const std::size_t *batched_row_cumsum_stride,
	float **batched_col_cumsum, const std::size_t *batched_col_cumsum_rows, const std::size_t *batched_col_cumsum_cols, const std::size_t *batched_col_cumsum_stride,
	float **batched_predicted_location, const std::size_t *batched_predicted_location_rows, const std::size_t *batched_predicted_location_cols, const std::size_t *batched_predicted_location_strides)
{

 }
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::select_most_probable_location(const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t *batched_location_probability_rows, const std::size_t *batched_location_probability_cols, const std::size_t *batched_location_probability_strides,
	float **batched_predicted_location, const std::size_t *batched_predicted_location_rows, const std::size_t *batched_predicted_location_cols, const std::size_t *batched_predicted_location_strides
)
{
#pragma omp parallel for
	for (int batch = 0; batch < batch_size; batch++)
	{
		const float *location_probability = batched_location_probability[batch];
		float *predicted_location = batched_predicted_location[batch];
		const std::size_t rows = batched_location_probability_rows[batch];
		const std::size_t stride = batched_location_probability_strides[batch];
		auto idx = cblas_isamax(rows * stride, location_probability, 1);
		std::size_t col = idx % stride;
		std::size_t row = idx / stride;
		float x = x_grid[col];
		float y = y_grid[row];

		predicted_location[0] = x;
		predicted_location[1] = y;
	}

}
#define PREFETCH_T0(addr,nrOfBytesAhead) _mm_prefetch(((char *)(addr))+nrOfBytesAhead,_MM_HINT_T0)
#define PREFETCH_T1(addr,nrOfBytesAhead) _mm_prefetch(((char *)(addr))+nrOfBytesAhead,_MM_HINT_T1)
template <TRN::CPU::Implementation Implementation>
static inline float  dot_product(const float *x, const float *a, const std::size_t &cols)
{
	std::size_t col = 0;
	auto y0 = setzero_ps();
	if (cols - col > _8)
	{
		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();
		auto y4 = setzero_ps();
		auto y5 = setzero_ps();
		auto y6 = setzero_ps();
		auto y7 = setzero_ps();

		for (; col + _8 - 1 < cols; col += _8)
		{
			y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
			y1 = mul_add_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), y1);
			y2 = mul_add_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2]), y2);
			y3 = mul_add_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3]), y3);
			y4 = mul_add_ps(load_ps(&a[col + _4]), load_ps(&x[col + _4]), y4);
			y5 = mul_add_ps(load_ps(&a[col + _5]), load_ps(&x[col + _5]), y5);
			y6 = mul_add_ps(load_ps(&a[col + _6]), load_ps(&x[col + _6]), y6);
			y7 = mul_add_ps(load_ps(&a[col + _7]), load_ps(&x[col + _7]), y7);
		}
		y0 = add_ps(y0, y1);
		y2 = add_ps(y2, y3);
		y4 = add_ps(y4, y5);
		y6 = add_ps(y6, y7);
		y0 = add_ps(y0, y2);
		y4 = add_ps(y4, y6);
		y0 = add_ps(y0, y4);
	}
	if (cols - col > _4)
	{
		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();
		for (; col + _4 - 1 < cols; col += _4)
		{
			y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
			y1 = mul_add_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), y1);
			y2 = mul_add_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2]), y2);
			y3 = mul_add_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3]), y3);
		}
		y0 = add_ps(y0, y1);
		y2 = add_ps(y2, y3);
		y0 = add_ps(y0, y2);
	}
	if (cols - col > _2)
	{
		auto y1 = setzero_ps();
		for (; col + _2 - 1 < cols; col += _2)
		{
			y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
			y1 = mul_add_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), y1);
		}
		y0 = add_ps(y0, y1);
	}
	if (cols - col > 0)
	{
		for (; col < cols; col += _1)
		{
			y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
		}
	}

	return hsum_ps(y0);
}


template <TRN::CPU::Implementation Implementation>
static inline void matrix_vector_product(const std::size_t &batch_size, 
	float **batched_a, const std::size_t *batched_a_rows, const std::size_t *batched_a_cols, const std::size_t *batched_a_strides,
	float **batched_x, const std::size_t *batched_x_rows, const std::size_t *batched_x_cols, const std::size_t *batched_x_strides,
	float **batched_y, const std::size_t *batched_y_rows, const std::size_t *batched_y_cols, const std::size_t *batched_y_strides)
{

	//A = a
	//B = x
	//c = y


	/*static const float alpha = 1.0f;
	static const float beta = 0.0f;
	static const std::size_t incX = 1;
	static const std::size_t incY = 1;

#pragma omp parallel for 
	for (int batch = 0; batch < batch_size; batch++)
	{
		const std::size_t M = batched_a_rows[batch];
		const std::size_t N = batched_a_cols[batch];
		const float *A = batched_a[batch];
		const float *X = batched_x[batch];
		float *Y = batched_y[batch];
		const std::size_t lda = batched_a_strides[batch];
		cblas_sgemv(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, M, N, alpha, A, lda, X, incX, beta, Y, incY);
	}*/

	std::size_t R = 0;
	for (std::size_t batch = 0; batch < batch_size; batch++)
		R += batched_a_rows[batch];

	std::size_t K = (R + _1 - 1) / _1;
#pragma omp parallel for schedule(static, batch_size)
	for (int k = 0; k < K; k++)
	{
		std::size_t batch = k % batch_size;
		std::size_t row = (k / batch_size) * _1;
		float *x = batched_x[batch];
		float *y = batched_y[batch];
		float *a = batched_a[batch];
		auto a_stride = batched_a_strides[batch];
		auto a_rows = batched_a_rows[batch];
		auto a_cols = batched_a_cols[batch];

		auto __dp = setzero_ps();
		for (std::size_t s = 0; s < _1; s++)
		{
			set_element(dot_product<Implementation>(x, &a[a_stride * (row + s)], a_cols), s, __dp);
		}
		stream_ps(&y[row], __dp);
	}
}

template <TRN::CPU::Implementation Implementation>
static inline void update_reservoir(
	const std::size_t &batch_size, const std::size_t &t,
	 float ** const batched_w_in, const std::size_t *batched_w_in_rows, const std::size_t *batched_w_in_cols, const  std::size_t *batched_w_in_strides,
	float ** const batched_x_in, const std::size_t *batched_x_in_rows, const std::size_t *batched_x_in_cols, const  std::size_t *batched_x_in_strides,
	float ** const batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const  std::size_t *batched_u_strides,
	float ** const batched_u_ffwd, const std::size_t *batched_u_ffwd_rows, const std::size_t *batched_u_ffwd_cols, const  std::size_t *batched_u_ffwd_strides,
	float ** const batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const  std::size_t *batched_p_strides,
	float ** const batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const  std::size_t *batched_x_res_strides,
	const typename TRN::CPU::Traits<Implementation>::type &leak_rate
)
{
	matrix_vector_product<Implementation>(batch_size,
		batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
		batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
		batched_u, batched_u_rows, batched_u_cols, batched_u_strides);

#pragma omp parallel for 
	for (int batch = 0; batch < batch_size; batch++)
	{
		auto rows = batched_w_in_rows[batch];
		auto u_ffwd = &batched_u_ffwd[batch][t * batched_u_ffwd_strides[batch]];
		auto u = batched_u[batch];
		auto p = batched_p[batch];
		auto x_res = batched_x_res[batch];
		auto x_in = batched_x_in[batch];
		auto w_in = batched_w_in[batch];
		auto w_in_stride = batched_w_in_strides[batch];

		std::size_t row = 0;
		if (rows - row > _8)
		{
			for (; row + _8 - 1 < rows; row += _8)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _0]), load_ps(&u_ffwd[row + _0])), __p0), __p0);
				stream_ps(&p[row + _0], __p0);
			
				auto __p1 = load_ps(&p[row + _1]);
				__p1 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _1]), load_ps(&u_ffwd[row + _1])), __p1), __p1);
				stream_ps(&p[row + _1], __p1);

				auto __p2 = load_ps(&p[row + _2]);
				__p2 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _2]), load_ps(&u_ffwd[row + _2])), __p2), __p2);
				stream_ps(&p[row + _2], __p2);

				auto __p3 = load_ps(&p[row + _3]);
				__p3 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _3]), load_ps(&u_ffwd[row + _3])), __p3), __p3);
				stream_ps(&p[row + _3], __p3);

				auto __p4 = load_ps(&p[row + _4]);
				__p4 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _4]), load_ps(&u_ffwd[row + _4])), __p4), __p4);
				stream_ps(&p[row + _4], __p4);

				auto __p5 = load_ps(&p[row + _5]);
				__p5 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _5]), load_ps(&u_ffwd[row + _5])), __p5), __p5);
				stream_ps(&p[row + _5], __p5);

				auto __p6 = load_ps(&p[row + _6]);
				__p6 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _6]), load_ps(&u_ffwd[row + _6])), __p6), __p6);
				stream_ps(&p[row + _6], __p6);

				auto __p7 = load_ps(&p[row + _7]);
				__p7 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _7]), load_ps(&u_ffwd[row + _7])), __p7), __p7);
				stream_ps(&p[row + _7], __p7);
			}
		}
		if (rows - row > _4)
		{
			for (; row + _4 - 1 < rows; row += _4)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _0]), load_ps(&u_ffwd[row + _0])), __p0), __p0);
				stream_ps(&p[row + _0], __p0);

				auto __p1 = load_ps(&p[row + _1]);
				__p1 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _1]), load_ps(&u_ffwd[row + _1])), __p1), __p1);
				stream_ps(&p[row + _1], __p1);

				auto __p2 = load_ps(&p[row + _2]);
				__p2 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _2]), load_ps(&u_ffwd[row + _2])), __p2), __p2);
				stream_ps(&p[row + _2], __p2);

				auto __p3 = load_ps(&p[row + _3]);
				__p3 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _3]), load_ps(&u_ffwd[row + _3])), __p3), __p3);
				stream_ps(&p[row + _3], __p3);
			}
		}
		if (rows - row > _2)
		{
			for (; row + _2 - 1 < rows; row += _2)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _0]), load_ps(&u_ffwd[row + _0])), __p0), __p0);
				stream_ps(&p[row + _0], __p0);

				auto __p1 = load_ps(&p[row + _1]);
				__p1 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _1]), load_ps(&u_ffwd[row + _1])), __p1), __p1);
				stream_ps(&p[row + _1], __p1);
			}
		}
		if (rows - row > 0)
		{
			for (; row + _1 - 1 < rows; row += _1)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(leak_rate, sub_ps(add_ps(load_ps(&u[row + _0]), load_ps(&u_ffwd[row + _0])), __p0), __p0);
				stream_ps(&p[row + _0], __p0);
			}
		}
		vsTanh(rows, p, x_res);
	}
}


template <TRN::CPU::Implementation Implementation>
static inline void update_reservoir_no_input(
	const std::size_t &batch_size, const std::size_t &t,
	float ** const batched_w_in, const std::size_t *batched_w_in_rows, const std::size_t *batched_w_in_cols, const  std::size_t *batched_w_in_strides,
	float ** const batched_x_in, const std::size_t *batched_x_in_rows, const std::size_t *batched_x_in_cols, const  std::size_t *batched_x_in_strides,
	float ** const batched_u, const std::size_t *batched_u_rows, const std::size_t *batched_u_cols, const  std::size_t *batched_u_strides,
	float ** const batched_p, const std::size_t *batched_p_rows, const std::size_t *batched_p_cols, const  std::size_t *batched_p_strides,
	float ** const batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const  std::size_t *batched_x_res_strides,
	const typename TRN::CPU::Traits<Implementation>::type &leak_rate
	)
{
	matrix_vector_product<Implementation>(batch_size,
		batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
		batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
		batched_u, batched_u_rows, batched_u_cols, batched_u_strides);

#pragma omp parallel for 
	for (int batch = 0; batch < batch_size; batch++)
	{
		auto rows = batched_w_in_rows[batch];
		auto u = batched_u[batch];
		auto p = batched_p[batch];
		auto x_res = batched_x_res[batch];
		auto x_in = batched_x_in[batch];
		auto w_in = batched_w_in[batch];
		auto w_in_stride = batched_w_in_strides[batch];

		std::size_t row = 0;
		if (rows - row > _8)
		{
			for (; row + _8 - 1 < rows; row += _8)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _0]), __p0), __p0);
				stream_ps(&p[row + _0], __p0);

				auto __p1 = load_ps(&p[row + _1]);
				__p1 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _1]), __p1), __p1);
				stream_ps(&p[row + _1], __p1);

				auto __p2 = load_ps(&p[row + _2]);
				__p2 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _2]), __p2), __p2);
				stream_ps(&p[row + _2], __p2);

				auto __p3 = load_ps(&p[row + _3]);
				__p3 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _3]), __p3), __p3);
				stream_ps(&p[row + _3], __p3);

				auto __p4 = load_ps(&p[row + _4]);
				__p4 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _4]), __p4), __p4);
				stream_ps(&p[row + _4], __p4);

				auto __p5 = load_ps(&p[row + _5]);
				__p5 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _5]), __p5), __p5);
				stream_ps(&p[row + _5], __p5);

				auto __p6 = load_ps(&p[row + _6]);
				__p6 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _6]), __p6), __p6);
				stream_ps(&p[row + _6], __p6);

				auto __p7 = load_ps(&p[row + _7]);
				__p7 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _7]), __p7), __p7);
				stream_ps(&p[row + _7], __p7);
			}
		}
		if (rows - row > _4)
		{
			for (; row + _4 - 1 < rows; row += _4)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _0]),  __p0), __p0);
				stream_ps(&p[row + _0], __p0);

				auto __p1 = load_ps(&p[row + _1]);
				__p1 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _1]),  __p1), __p1);
				stream_ps(&p[row + _1], __p1);

				auto __p2 = load_ps(&p[row + _2]);
				__p2 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _2]),  __p2), __p2);
				stream_ps(&p[row + _2], __p2);

				auto __p3 = load_ps(&p[row + _3]);
				__p3 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _3]), __p3), __p3);
				stream_ps(&p[row + _3], __p3);
			}
		}
		if (rows - row > _2)
		{
			for (; row + _2 - 1 < rows; row += _2)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _0]),  __p0), __p0);
				stream_ps(&p[row + _0], __p0);

				auto __p1 = load_ps(&p[row + _1]);
				__p1 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _1]), __p1), __p1);
				stream_ps(&p[row + _1], __p1);
			}
		}
		if (rows - row > 0)
		{
			for (; row + _1 - 1 < rows; row += _1)
			{
				auto __p0 = load_ps(&p[row + _0]);
				__p0 = mul_add_ps(leak_rate, sub_ps(load_ps(&u[row + _0]), __p0), __p0);
				stream_ps(&p[row + _0], __p0);
			}
		}
		vsTanh(rows, p, x_res);
	}
}

template <TRN::CPU::Implementation Implementation>
static inline void premultiply(
	const std::size_t &batch_size,
	const float **batched_incoming, const std::size_t *incoming_rows, const std::size_t *incoming_cols, const std::size_t *incoming_strides,
	const float **batched_w_ffwd, const std::size_t *w_ffwd_rows, const std::size_t *w_ffwd_cols, const std::size_t *w_ffwd_strides,

	float **batched_u_ffwd, const std::size_t *u_ffwd_rows, const std::size_t *u_ffwd_cols, const std::size_t *u_ffwd_strides)
{


	std::vector<CBLAS_TRANSPOSE> TransA_array(batch_size);
	std::vector<CBLAS_TRANSPOSE> TransB_array(batch_size);
	std::vector<std::size_t> group_size(batch_size);
	std::vector<float>  alpha_array(batch_size);
	std::vector<float> beta_array(batch_size);
	const std::size_t *M_array = u_ffwd_rows;
	const std::size_t *N_array = u_ffwd_cols;
	const std::size_t *K_array = incoming_cols;
	const std::size_t *lda_array = incoming_strides;
	const std::size_t *ldb_array = w_ffwd_strides;
	const std::size_t *ldc_array = u_ffwd_strides;
	const float **A_array = batched_incoming;
	const float **B_array = batched_w_ffwd;
	float **C_array = batched_u_ffwd;

	std::size_t group_count = batch_size;



	std::fill(TransA_array.begin(), TransA_array.end(), CBLAS_TRANSPOSE::CblasNoTrans);
	std::fill(TransB_array.begin(), TransB_array.end(), CBLAS_TRANSPOSE::CblasTrans);
	std::fill(group_size.begin(), group_size.end(), 1);
	std::fill(alpha_array.begin(), alpha_array.end(), 1.0f);
	std::fill(beta_array.begin(), beta_array.end(), 0.0f);


	cblas_sgemm_batch
	(
		CBLAS_LAYOUT::CblasRowMajor,
		TransA_array.data(), TransB_array.data(),
		M_array, N_array, K_array, alpha_array.data(), A_array, lda_array, B_array, ldb_array, beta_array.data(), C_array, ldc_array, group_count, group_size.data()
	);
}



template <TRN::CPU::Implementation Implementation>
static inline void batched_update_readout_activation(const std::size_t &batch_size,
	float **batched_x, const std::size_t *batched_x_rows, const std::size_t *batched_x_cols, const std::size_t *batched_x_strides)
{


	std::size_t K = batch_size;
#pragma omp parallel for schedule(static, batch_size)
	for (int batch = 0; batch < batch_size; batch++)
	{
	
		float *x = batched_x[batch];
		auto cols = batched_x_cols[batch];
		vsTanh(cols, x, x);
	}
}



template <TRN::CPU::Implementation Implementation, typename Parameter>
static inline void batched_update_readout(
	const std::size_t &batch_size, const std::size_t & t, const Parameter &parameter,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides)
{
	matrix_vector_product<Implementation>(
		batch_size,
		batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
		batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
		batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides);
	batched_update_readout_activation<Implementation>
		(
			batch_size,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides
			);
}



template <TRN::CPU::Implementation Implementation>
static inline void batched_update_readout(
	const std::size_t &batch_size, const std::size_t & t, const Widrow_Hoff &parameter,
	float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_strides,
	float **batched_w_ro, const std::size_t *batched_w_ro_rows, const std::size_t *batched_w_ro_cols, const std::size_t *batched_w_ro_strides)
{
	matrix_vector_product<Implementation>(
		batch_size,
		batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
		batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
		batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides);
	batched_update_readout_activation<Implementation>
		(
			batch_size,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides
			);
	const float learning_rate = parameter.get_learning_rate();
	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		const std::size_t rows = batched_w_ro_rows[batch];
		const std::size_t cols = batched_w_ro_cols[batch];
		std::size_t w_ro_stride = batched_w_ro_strides[batch];
		std::size_t expected_stride = batched_expected_strides[batch];
		float *w_ro = batched_w_ro[batch];
		float *x_ro = batched_x_ro[batch];
		float *expected = &batched_expected[batch][t * expected_stride];
		float *error = batched_error[batch];
		float *x_res = batched_x_res[batch];
		
#pragma omp parallel for
		for (int row = 0; row < rows; row++)
		{
			float *w_ro_row = &w_ro[row  * w_ro_stride];
			const auto x_ro_row = x_ro[row];
			const auto post_error = learning_rate * (expected[row] - x_ro_row) * (1.0f - x_ro_row * x_ro_row);
			const auto __post_error = set1_ps(post_error);
			error[row] = post_error;
			std::size_t col = 0;

			if (cols - col > _8)
			{
				for (; col + _8 - 1 < cols; col += _8)
				{
					stream_ps(&w_ro_row[col + _0], mul_add_ps(__post_error, load_ps(&x_res[col + _0]), load_ps(&w_ro_row[col + _0])));
					stream_ps(&w_ro_row[col + _1], mul_add_ps(__post_error, load_ps(&x_res[col + _1]), load_ps(&w_ro_row[col + _1])));
					stream_ps(&w_ro_row[col + _2], mul_add_ps(__post_error, load_ps(&x_res[col + _2]), load_ps(&w_ro_row[col + _2])));
					stream_ps(&w_ro_row[col + _3], mul_add_ps(__post_error, load_ps(&x_res[col + _3]), load_ps(&w_ro_row[col + _3])));
					stream_ps(&w_ro_row[col + _4], mul_add_ps(__post_error, load_ps(&x_res[col + _4]), load_ps(&w_ro_row[col + _4])));
					stream_ps(&w_ro_row[col + _5], mul_add_ps(__post_error, load_ps(&x_res[col + _5]), load_ps(&w_ro_row[col + _5])));
					stream_ps(&w_ro_row[col + _6], mul_add_ps(__post_error, load_ps(&x_res[col + _6]), load_ps(&w_ro_row[col + _6])));
					stream_ps(&w_ro_row[col + _7], mul_add_ps(__post_error, load_ps(&x_res[col + _7]), load_ps(&w_ro_row[col + _7])));
				}
			}
			if (cols - col > _4)
			{
				for (; col + _4 - 1 < cols; col += _4)
				{
					stream_ps(&w_ro_row[col + _0], mul_add_ps(__post_error, load_ps(&x_res[col + _0]), load_ps(&w_ro_row[col + _0])));
					stream_ps(&w_ro_row[col + _1], mul_add_ps(__post_error, load_ps(&x_res[col + _1]), load_ps(&w_ro_row[col + _1])));
					stream_ps(&w_ro_row[col + _2], mul_add_ps(__post_error, load_ps(&x_res[col + _2]), load_ps(&w_ro_row[col + _2])));
					stream_ps(&w_ro_row[col + _3], mul_add_ps(__post_error, load_ps(&x_res[col + _3]), load_ps(&w_ro_row[col + _3])));
				}
			}
			if (cols - col > _2)
			{
				for (; col + _2 - 1 < cols; col += _2)
				{
					stream_ps(&w_ro_row[col + _0], mul_add_ps(__post_error, load_ps(&x_res[col + _0]), load_ps(&w_ro_row[col + _0])));
					stream_ps(&w_ro_row[col + _1], mul_add_ps(__post_error, load_ps(&x_res[col + _1]), load_ps(&w_ro_row[col + _1])));
				}
			}
			if (cols - col > 0)
			{
				for (; col + _1 - 1 < cols; col += _1)
				{
					stream_ps(&w_ro_row[col + _0], mul_add_ps(__post_error, load_ps(&x_res[col + _0]), load_ps(&w_ro_row[col + _0])));
				}
			}
		}
	}
}


template <bool gather_states>
static void copy_states(
 const std::size_t &batch_size, const std::size_t &t, const std::size_t &ts,
	const std::size_t &stimulus_size,
	const std::size_t &reservoir_size,
	const std::size_t &prediction_size,
	const std::size_t &stimulus_stride,
	const std::size_t &reservoir_stride,
	const std::size_t &prediction_stride,
	const float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	const float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	const float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	const float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float *states, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{}

template <>
static void copy_states<true>(const std::size_t &batch_size, const std::size_t &t, const std::size_t &ts,
	const std::size_t &stimulus_size,
	const std::size_t &reservoir_size,
	const std::size_t &prediction_size,
	const std::size_t &stimulus_stride,
	const std::size_t &reservoir_stride,
	const std::size_t &prediction_stride,
	const float **batched_incoming, const std::size_t *batched_incoming_rows, const std::size_t *batched_incoming_cols, const std::size_t *batched_incoming_strides,
	const float **batched_expected, const std::size_t *batched_expected_rows, const std::size_t *batched_expected_cols, const std::size_t *batched_expected_strides,
	const float **batched_x_ro, const std::size_t *batched_x_ro_rows, const std::size_t *batched_x_ro_cols, const std::size_t *batched_x_ro_strides,
	const float **batched_x_res, const std::size_t *batched_x_res_rows, const std::size_t *batched_x_res_cols, const std::size_t *batched_x_res_strides,
	float *states, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{
	std::size_t offset = 0;
	float *states_ts = &states[ts * states_stride];

	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		std::size_t offset = 0;
		std::size_t  stimulus_col = batch * stimulus_stride + batch_size * offset;
		std::memcpy(states_ts + stimulus_col, &batched_incoming[batch][t * batched_incoming_strides[batch]], sizeof(float) * stimulus_size);
		offset += stimulus_stride;

		std::size_t  desired_col = batch * prediction_stride + batch_size * offset;
		std::memcpy(states_ts + desired_col, &batched_expected[batch][t * batched_expected_strides[batch]], sizeof(float) * prediction_size);
		offset += prediction_stride;

		std::size_t  reservoir_col = batch * reservoir_stride + batch_size * offset;
		std::memcpy(states_ts + reservoir_col, batched_x_res[batch], sizeof(float) * reservoir_size);
		offset += reservoir_stride;

		std::size_t  predicted_col = batch * prediction_stride + batch_size * offset;
		std::memcpy(states_ts + predicted_col, batched_x_ro[batch], sizeof(float) * prediction_size);
		offset += prediction_stride;
	}
}

template <TRN::CPU::Implementation Implementation, bool overwrite_states>
struct initialize_states
{
	void operator () (const std::size_t &batch_size, unsigned long &seed, float **batched_ptr, const std::size_t *batched_rows, const std::size_t *batched_cols, const std::size_t *batched_strides, const float &initial_state_scale)
	{

	}
};

template <TRN::CPU::Implementation Implementation>
struct initialize_states<Implementation, true>
{
	void operator () (const std::size_t &batch_size, unsigned long &seed, float **batched_ptr, const std::size_t *batched_rows, const std::size_t *batched_cols, const std::size_t *batched_strides, const float &initial_state_scale)
	{
		TRN::CPU::Random::uniform_implementation(seed, batched_ptr, batch_size, batched_rows, batched_cols, batched_strides, false, -initial_state_scale, initial_state_scale, 0.0f);
	
		for (int k = 0; k < batch_size; k++)
			seed += batch_size * batched_rows[k] * batched_cols[k];
	}
};

template<TRN::CPU::Implementation Implementation, bool gather_states, bool overwrite_states, typename Parameter>
static inline void update_model(
	const std::size_t &batch_size,
	unsigned long &seed,
	const Parameter &parameter,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
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
	float *c;
	std::size_t c_stride;



	auto __leak_rate = set1_ps(leak_rate);


	static initialize_states<Implementation, overwrite_states> initializer;




	premultiply<Implementation>(
		batch_size,
		(const float **)batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
		(const float **)batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
		batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides);

	std::size_t ts = 0;
	for (std::size_t repetition = 0; repetition < repetitions; repetition++)
	{
		initializer(batch_size, seed, batched_p, batched_p_rows, batched_p_cols, batched_p_strides, initial_state_scale);
		initializer(batch_size, seed, batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides, initial_state_scale);
	
		for (std::size_t k = 0; k < durations[repetition]; k++, ts++)
		{
			int t = offsets[ts];
	
			if (t < 0)
			{
				t = -t;
				update_reservoir_no_input<Implementation>(batch_size, t,
					batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
					batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
					batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
					batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
					batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides, __leak_rate);
			
			}
			else
			{
				update_reservoir<Implementation>(batch_size, t,
					batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
					batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
					batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
					batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
					batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
					batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides, __leak_rate);
			}
		
			batched_update_readout<Implementation>(batch_size, t, parameter,
				batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
				batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
				batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
				batched_error, batched_error_rows, batched_error_cols, batched_error_strides,
				batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides
				);
			//INFORMATION d ;
			copy_states<gather_states>(batch_size, t, ts,
				stimulus_size, reservoir_size, prediction_size,
				stimulus_stride, reservoir_stride, prediction_stride,
				(const float **)batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
				(const float **)batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
				(const float **)batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
				(const float **)batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
				states_samples, states_rows, states_cols, states_stride
				);
		}
	}
	
}
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::learn_widrow_hoff(
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

	const float &learning_rate)
{
	if (states_samples == NULL)
	{
		update_model<Implementation, false, true>(
			batch_size, seed, Widrow_Hoff(learning_rate),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
			batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			batched_error, batched_error_rows, batched_error_cols, batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride
			);
	}
	else
	{
		update_model<Implementation, true, true>(
			batch_size, seed, Widrow_Hoff(learning_rate),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
			batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			batched_error, batched_error_rows, batched_error_cols, batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride
			);
	}
}
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::prime(
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
		update_model<Implementation, false, true>(
			batch_size, seed, Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
			batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			batched_error, batched_error_rows, batched_error_cols, batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride);
	}
	else
	{
		update_model<Implementation, true, true>(
			batch_size, seed, Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
			batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			batched_error, batched_error_rows, batched_error_cols, batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride);
	}

}

static std::mutex mutex;

template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::generate(
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
		update_model<Implementation, false, false>(
			batch_size, seed, Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
			batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			batched_error, batched_error_rows, batched_error_cols, batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride);
	}
	else
	{
		update_model<Implementation, true, false>(
			batch_size, seed, Nothing(),
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size,
			leak_rate, initial_state_scale,
			batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
			batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
			batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
			batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
			batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
			batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
			batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
			batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
			batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
			batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
			batched_error, batched_error_rows, batched_error_cols, batched_error_strides,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride);
	}

}


template <TRN::CPU::Implementation Implementation>
std::shared_ptr<TRN::CPU::Algorithm<Implementation>> TRN::CPU::Algorithm<Implementation>::create()
{
	return std::make_shared<TRN::CPU::Algorithm<Implementation>>();
}





