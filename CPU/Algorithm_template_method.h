#pragma once

#include "Implementation.h"
#include "Algorithm_impl.h"
#include "Memory.h"
#include "Random.h"
#include <ctime>
#include <vector>
#include <opencv2/core.hpp>



enum LearningRule
{
	WIDROW_HOFF,
	NOTHING
};

template <TRN::CPU::Implementation Implementation>
TRN::CPU::Algorithm<Implementation>::Algorithm() :
	handle(std::make_unique<Handle<Implementation>>())
{

}

template <TRN::CPU::Implementation Implementation>
TRN::CPU::Algorithm<Implementation>::~Algorithm(
{
	handle.reset();
}
template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>:preallocate(const std::size_t &stimulus_size, const std::size_t &reservoir_size,
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
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			auto _d0 = sub_ps(load_ps(&A[col + _0]), load_ps(&B[col + _0]));
			acc = mul_add_ps(_d0, _d0, acc);
		}
	}

	 return (hsum_ps(acc) / cols);
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
/*#pragma omp parallel for
	for (int row = 0; row < predicted_rows; row++)
	{
		result[row * result_stride] = compute_mse<Implementation>(&predicted[row * predicted_stride], &expected[row * expected_stride], expected_cols);
	}*/
}

template <TRN::CPU::Implementation Implementation>
static inline void 	weighted_sum(
	const float *A, const typename TRN::CPU::Traits<Implementation>::type  &a,
	const float *B, const typename TRN::CPU::Traits<Implementation>::type  &b,
	const std::size_t &cols, float *C)
{
	std::size_t col = 0;
	/*if (cols - col > 06)
	{
		for (; col + _16 - 1 < cols; col += _16)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, mul_ps(load_ps(&B[col + _1]), b)));
			stream_ps(&C[col + _2], mul_add_ps(load_ps(&A[col + _2]), a, mul_ps(load_ps(&B[col + _2]), b)));
			stream_ps(&C[col + _3], mul_add_ps(load_ps(&A[col + _3]), a, mul_ps(load_ps(&B[col + _3]), b)));
			stream_ps(&C[col + _4], mul_add_ps(load_ps(&A[col + _4]), a, mul_ps(load_ps(&B[col + _4]), b)));
			stream_ps(&C[col + _5], mul_add_ps(load_ps(&A[col + _5]), a, mul_ps(load_ps(&B[col + _5]), b)));
			stream_ps(&C[col + _6], mul_add_ps(load_ps(&A[col + _6]), a, mul_ps(load_ps(&B[col + _6]), b)));
			stream_ps(&C[col + _7], mul_add_ps(load_ps(&A[col + _7]), a, mul_ps(load_ps(&B[col + _7]), b)));
			stream_ps(&C[col + _8], mul_add_ps(load_ps(&A[col + _8]), a, mul_ps(load_ps(&B[col + _8]), b)));
			stream_ps(&C[col + _9], mul_add_ps(load_ps(&A[col + _1]), a, mul_ps(load_ps(&B[col + _1]), b)));
			stream_ps(&C[col + _10], mul_add_ps(load_ps(&A[col + _10]), a, mul_ps(load_ps(&B[col + _10]), b)));
			stream_ps(&C[col + _11], mul_add_ps(load_ps(&A[col + _11]), a, mul_ps(load_ps(&B[col + _11]), b)));
			stream_ps(&C[col + _12], mul_add_ps(load_ps(&A[col + _12]), a, mul_ps(load_ps(&B[col + _12]), b)));
			stream_ps(&C[col + _13], mul_add_ps(load_ps(&A[col + _13]), a, mul_ps(load_ps(&B[col + _13]), b)));
			stream_ps(&C[col + _14], mul_add_ps(load_ps(&A[col + _14]), a, mul_ps(load_ps(&B[col + _14]), b)));
			stream_ps(&C[col + _15], mul_add_ps(load_ps(&A[col + _15]), a, mul_ps(load_ps(&B[col + _15]), b)));
		}
	}*/
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
		for (; col  < cols; col += _1)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, mul_ps(load_ps(&B[col + _0]), b)));
		}
	}
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
	/*if (cols - col > 06)
	{
		for (; col + _16 - 1 < cols; col += _16)
		{
			stream_ps(&C[col + _0], mul_add_ps(load_ps(&A[col + _0]), a, load_ps(&C[col + _0])));
			stream_ps(&C[col + _1], mul_add_ps(load_ps(&A[col + _1]), a, load_ps(&C[col + _1])));
			stream_ps(&C[col + _2], mul_add_ps(load_ps(&A[col + _2]), a, load_ps(&C[col + _2])));
			stream_ps(&C[col + _3], mul_add_ps(load_ps(&A[col + _3]), a, load_ps(&C[col + _3])));
			stream_ps(&C[col + _4], mul_add_ps(load_ps(&A[col + _4]), a, load_ps(&C[col + _4])));
			stream_ps(&C[col + _5], mul_add_ps(load_ps(&A[col + _5]), a, load_ps(&C[col + _5])));
			stream_ps(&C[col + _6], mul_add_ps(load_ps(&A[col + _6]), a, load_ps(&C[col + _6])));
			stream_ps(&C[col + _7], mul_add_ps(load_ps(&A[col + _7]), a, load_ps(&C[col + _7])));
			stream_ps(&C[col + _8], mul_add_ps(load_ps(&A[col + _8]), a, load_ps(&C[col + _8])));
			stream_ps(&C[col + _9], mul_add_ps(load_ps(&A[col + _9]), a, load_ps(&C[col + _9])));
			stream_ps(&C[col + _10], mul_add_ps(load_ps(&A[col + _10]), a, load_ps(&C[col + _10])));
			stream_ps(&C[col + _11], mul_add_ps(load_ps(&A[col + _11]), a, load_ps(&C[col + _11])));
			stream_ps(&C[col + _12], mul_add_ps(load_ps(&A[col + _12]), a, load_ps(&C[col + _12])));
			stream_ps(&C[col + _13], mul_add_ps(load_ps(&A[col + _13]), a, load_ps(&C[col + _13])));
			stream_ps(&C[col + _14], mul_add_ps(load_ps(&A[col + _14]), a, load_ps(&C[col + _14])));
			stream_ps(&C[col + _15], mul_add_ps(load_ps(&A[col + _15]), a, load_ps(&C[col + _15])));
		}
	}*/
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
	/*if (cols - col > 06)
	{
		for (; col + _16 - 1 < cols; col += _16)
		{
			stream_ps(&grid_centered2[col + _0], sqr_ps(sub_ps(load_ps(&grid[col + _0]), current)));
			stream_ps(&grid_centered2[col + _1], sqr_ps(sub_ps(load_ps(&grid[col + _1]), current)));
			stream_ps(&grid_centered2[col + _2], sqr_ps(sub_ps(load_ps(&grid[col + _2]), current)));
			stream_ps(&grid_centered2[col + _3], sqr_ps(sub_ps(load_ps(&grid[col + _3]), current)));
			stream_ps(&grid_centered2[col + _4], sqr_ps(sub_ps(load_ps(&grid[col + _4]), current)));
			stream_ps(&grid_centered2[col + _5], sqr_ps(sub_ps(load_ps(&grid[col + _5]), current)));
			stream_ps(&grid_centered2[col + _6], sqr_ps(sub_ps(load_ps(&grid[col + _6]), current)));
			stream_ps(&grid_centered2[col + _7], sqr_ps(sub_ps(load_ps(&grid[col + _7]), current)));
			stream_ps(&grid_centered2[col + _8], sqr_ps(sub_ps(load_ps(&grid[col + _8]), current)));
			stream_ps(&grid_centered2[col + _9], sqr_ps(sub_ps(load_ps(&grid[col + _9]), current)));
			stream_ps(&grid_centered2[col + _10], sqr_ps(sub_ps(load_ps(&grid[col + _10]), current)));
			stream_ps(&grid_centered2[col + _11], sqr_ps(sub_ps(load_ps(&grid[col + _11]), current)));
			stream_ps(&grid_centered2[col + _12], sqr_ps(sub_ps(load_ps(&grid[col + _12]), current)));
			stream_ps(&grid_centered2[col + _13], sqr_ps(sub_ps(load_ps(&grid[col + _13]), current)));
			stream_ps(&grid_centered2[col + _14], sqr_ps(sub_ps(load_ps(&grid[col + _14]), current)));
			stream_ps(&grid_centered2[col + _15], sqr_ps(sub_ps(load_ps(&grid[col + _15]), current)));
		}
	}*/
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
static inline void 	circle(const float *x_grid_centered2, const std::size_t &cols, const typename TRN::CPU::Traits<Implementation>::type &y2, const typename TRN::CPU::Traits<Implementation>::type &r2, float *location_probability_row)
{
	const auto __zero = setzero_ps();
	std::size_t col = 0;
	/*if (cols - col > 06)
	{
		for (; col + _16 - 1 < cols; col += _16)
		{
			stream_ps(&location_probability_row[col + _0], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _0]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _0]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _1], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _1]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _1]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _2], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _2]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _2]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _3], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _3]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _3]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _4], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _4]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _4]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _5], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _5]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _5]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _6], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _6]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _6]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _7], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _7]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _7]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _8], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _8]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _8]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _9], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _9]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _9]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _10], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _10]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _10]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _11], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _11]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _11]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _12], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _12]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _12]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _13], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _13]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _13]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _14], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _14]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _14]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _15], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _15]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _15]), y2), r2, _CMP_LT_OQ)));
		}
	}*/
	if (cols - col > _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			stream_ps(&location_probability_row[col + _0], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _0]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _0]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _1], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _1]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _1]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _2], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _2]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _2]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _3], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _3]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _3]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _4], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _4]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _4]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _5], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _5]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _5]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _6], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _6]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _6]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _7], blendv_ps<Implementation>(__zero, load_ps(&location_probability_row[col + _7]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _7]), y2), r2, _CMP_LT_OQ)));
		}
	}
	if (cols - col > _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, load_ps(&location_probability_row[col + _0]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _0]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _1], blendv_ps(__zero, load_ps(&location_probability_row[col + _1]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _1]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _2], blendv_ps(__zero, load_ps(&location_probability_row[col + _2]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _2]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _3], blendv_ps(__zero, load_ps(&location_probability_row[col + _3]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _3]), y2), r2, _CMP_LT_OQ)));
		}
	}
	if (cols - col > _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, load_ps(&location_probability_row[col + _0]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _0]), y2), r2, _CMP_LT_OQ)));
			stream_ps(&location_probability_row[col + _1], blendv_ps(__zero, load_ps(&location_probability_row[col + _1]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _1]), y2), r2, _CMP_LT_OQ)));
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			stream_ps(&location_probability_row[col + _0], blendv_ps(__zero, load_ps(&location_probability_row[col + _0]), cmp_ps(add_ps(load_ps(&x_grid_centered2[col + _0]), y2), r2, _CMP_LT_OQ)));
		}
	}
}

template <TRN::CPU::Implementation Implementation>
static inline float accumulate_ps(const float *firing_rate_row, const std::size_t &cols,
	const typename TRN::CPU::Traits<Implementation>::type  &__prediction,
	const typename TRN::CPU::Traits<Implementation>::type  &__inv_sigma2,
	float *hypothesis_row)
{
	auto __acc = setzero_ps();
	std::size_t col = 0;
	/*if (cols - col > 06)
	{
		for (; col + _16 - 1 < cols; col += _16)
		{
			auto __h0 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2));
			__acc = add_ps(__h0, __acc);
			stream_ps(&hypothesis_row[col + _0], __h0);
			auto __h1 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _1]))), __inv_sigma2));
			__acc = add_ps(__h1, __acc);
			stream_ps(&hypothesis_row[col + _1], __h1);
			auto __h2 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _2]))), __inv_sigma2));
			__acc = add_ps(__h2, __acc);
			stream_ps(&hypothesis_row[col + _2], __h2);
			auto __h3 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _3]))), __inv_sigma2));
			__acc = add_ps(__h3, __acc);
			stream_ps(&hypothesis_row[col + _3], __h3);
			auto __h4 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _4]))), __inv_sigma2));
			__acc = add_ps(__h4, __acc);
			stream_ps(&hypothesis_row[col + _4], __h4);
			auto __h5 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _5]))), __inv_sigma2));
			__acc = add_ps(__h5, __acc);
			stream_ps(&hypothesis_row[col + _5], __h5);
			auto __h6 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _6]))), __inv_sigma2));
			__acc = add_ps(__h6, __acc);
			stream_ps(&hypothesis_row[col + _6], __h6);
			auto __h7 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _7]))), __inv_sigma2));
			__acc = add_ps(__h7, __acc);
			stream_ps(&hypothesis_row[col + _7], __h7);
			auto __h8 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _8]))), __inv_sigma2));
			__acc = add_ps(__h8, __acc);
			stream_ps(&hypothesis_row[col + _8], __h8);
			auto __h9 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _9]))), __inv_sigma2));
			__acc = add_ps(__h9, __acc);
			stream_ps(&hypothesis_row[col + _9], __h9);
			auto __h10 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _10]))), __inv_sigma2));
			__acc = add_ps(__h10, __acc);
			stream_ps(&hypothesis_row[col + _10], __h10);
			auto __h11 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _11]))), __inv_sigma2));
			__acc = add_ps(__h11, __acc);
			stream_ps(&hypothesis_row[col + _11], __h11);
			auto __h12 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _12]))), __inv_sigma2));
			__acc = add_ps(__h12, __acc);
			stream_ps(&hypothesis_row[col + _12], __h12);
			auto __h13 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _13]))), __inv_sigma2));
			__acc = add_ps(__h13, __acc);
			stream_ps(&hypothesis_row[col + _13], __h13);
			auto __h14 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _14]))), __inv_sigma2));
			__acc = add_ps(__h14, __acc);
			stream_ps(&hypothesis_row[col + _14], __h4);
			auto __h15 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _15]))), __inv_sigma2));
			__acc = add_ps(__h15, __acc);
			stream_ps(&hypothesis_row[col + _15], __h15);
		}
	}*/
	if (cols - col > _8)
	{
		for (; col + _8 - 1 < cols; col += _8)
		{
			auto __h0 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2));
			__acc = add_ps(__h0, __acc);
			stream_ps(&hypothesis_row[col + _0], __h0);
			auto __h1 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _1]))), __inv_sigma2));
			__acc = add_ps(__h1, __acc);
			stream_ps(&hypothesis_row[col + _1], __h1);
			auto __h2 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _2]))), __inv_sigma2));
			__acc = add_ps(__h2, __acc);
			stream_ps(&hypothesis_row[col + _2], __h2);
			auto __h3 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _3]))), __inv_sigma2));
			__acc = add_ps(__h3, __acc);
			stream_ps(&hypothesis_row[col + _3], __h3);			
			auto __h4 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _4]))), __inv_sigma2));
			__acc = add_ps(__h4, __acc);
			stream_ps(&hypothesis_row[col + _4], __h4);
			auto __h5 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _5]))), __inv_sigma2));
			__acc = add_ps(__h5, __acc);
			stream_ps(&hypothesis_row[col + _5], __h5);
			auto __h6 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _6]))), __inv_sigma2));
			__acc = add_ps(__h6, __acc);
			stream_ps(&hypothesis_row[col + _6], __h6);
			auto __h7 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _7]))), __inv_sigma2));
			__acc = add_ps(__h7, __acc);
			stream_ps(&hypothesis_row[col + _7], __h7);
		}
	}
	if (cols - col > _4)
	{
		for (; col + _4 - 1 < cols; col += _4)
		{
			auto __h0 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2));
			__acc = add_ps(__h0, __acc);
			stream_ps(&hypothesis_row[col + _0], __h0);
			auto __h1 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _1]))), __inv_sigma2));
			__acc = add_ps(__h1, __acc);
			stream_ps(&hypothesis_row[col + _1], __h1);
			auto __h2 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _2]))), __inv_sigma2));
			__acc = add_ps(__h2, __acc);
			stream_ps(&hypothesis_row[col + _2], __h2);
			auto __h3 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _3]))), __inv_sigma2));
			__acc = add_ps(__h3, __acc);
			stream_ps(&hypothesis_row[col + _3], __h3);
		}
	}
	if (cols - col > _2)
	{
		for (; col + _2 - 1 < cols; col += _2)
		{
			auto __h0 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2));
			__acc = add_ps(__h0, __acc);
			stream_ps(&hypothesis_row[col + _0], __h0);
			auto __h1 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _1]))), __inv_sigma2));
			__acc = add_ps(__h1, __acc);
			stream_ps(&hypothesis_row[col + _1], __h1);
		}
	}
	 if (cols - col > 0)
	{
		for (; col  < cols; col += _1)
		{
			auto __h0 = exp_ps(mul_ps(sqr_ps(sub_ps(__prediction, load_ps(&firing_rate_row[col + _0]))), __inv_sigma2));
			__acc = add_ps(__h0, __acc);
			stream_ps(&hypothesis_row[col + _0], __h0);
		}
	}


	return hsum_ps(__acc);
}

template <TRN::CPU::Implementation Implementation>
void TRN::CPU::Algorithm<Implementation>::place_cell_location_probability(
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float &sigma,
	const float &radius,
	const float ** firing_rate_map, const std::size_t *firing_rate_map_rows, const std::size_t *firing_rate_map_cols, const std::size_t *firing_rate_map_strides,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	float **x_grid_centered2, const std::size_t *x_grid_centered2_rows, const std::size_t *x_grid_centered2_cols, const std::size_t *x_grid_centered2_strides,
	float **y_grid_centered2, const std::size_t *y_grid_centered2_rows, const std::size_t *y_grid_centered2_cols, const std::size_t *y_grid_centered2_strides,
	float **scale, const std::size_t *scale_rows, const std::size_t *scale_cols, const std::size_t *scale_strides,
	const float **prediction, const std::size_t *prediction_rows, const std::size_t *prediction_cols, const std::size_t *prediction_strides,
	float *** hypothesis_map, const std::size_t **hypothesis_map_rows, const std::size_t **hypothesis_map_cols, const std::size_t **hypothesis_map_strides,
	float ** location_probability, const std::size_t *location_probability_rows, const std::size_t *location_probability_cols, const std::size_t *location_probability_strides,
	float ** predicted_position, const std::size_t *predicted_position_rows, const std::size_t *predicted_position_cols, const std::size_t *predicted_position_strides)
{

	//	std::cout << __FUNCTION__ << " current position " << x_current << ", " << y_current << std::endl;

		/*double duration;

		const float _inv_sigma2 = -1.0f / (sigma*sigma);
		const auto ___inv_sigma2 = set1_ps(_inv_sigma2);
		const auto __x_current = set1_ps(x_current);
		const auto __y_current = set1_ps(y_current);
		const auto r2 = radius * radius;
		const auto __r2 = set1_ps(r2);
	#pragma omp parallel sections
		{
	#pragma omp section
			{
				diff_square<Implementation>(__x_current, x_grid, x_size, x_grid_centered2);
			}
	#pragma omp section
			{
				diff_square<Implementation>(__y_current, y_grid, y_size, y_grid_centered2);
			}
		}

	#pragma omp parallel for
		for (int k = 0; k < place_cells_number; k++)
		{
			auto firing_rate_k = &firing_rate_map[k * firing_rate_map_stride * y_size];
			auto hypothesis_k = &hypothesis_map[k *  hypothesis_map_stride * y_size];
			float sum = 0.0f;
			const float &p = prediction[k];
			const auto &__prediction = set1_ps(p);

			for (std::size_t row = 0; row < y_size; row++)
			{
				auto firing_rate_row = &firing_rate_k[row * firing_rate_map_stride];
				auto hypothesis_row = &hypothesis_k[row * hypothesis_map_stride];

				sum += accumulate_ps<Implementation>(firing_rate_row, hypothesis_map_stride, __prediction, ___inv_sigma2, hypothesis_row);
			}
			scale[k] = 1.0f / (sum * (float)place_cells_number);
		}

		const std::size_t place_cells_number_range = place_cells_number / 2;
		const std::size_t place_cells_number_remaining = place_cells_number - place_cells_number_range * 2;

	#pragma omp parallel for
		for (int k = 0; k < place_cells_number_range; k++)
		{
			auto a = (k);
			auto b = (place_cells_number_range + k);
			auto hypothesis_a = &hypothesis_map[a * hypothesis_map_stride * y_size];
			auto hypothesis_b = &hypothesis_map[b * hypothesis_map_stride * y_size];
			const auto scale_a = set1_ps(scale[a]);
			const auto scale_b = set1_ps(scale[b]);
			for (std::size_t row = 0; row < y_size; row++)
			{
				auto hypothesis_a_row = &hypothesis_a[row * hypothesis_map_stride];
				auto hypothesis_b_row = &hypothesis_b[row* hypothesis_map_stride];
				weighted_sum<Implementation>(hypothesis_a_row, scale_a, hypothesis_b_row, scale_b, x_size, hypothesis_a_row);
			}
		}

		if (place_cells_number_range >= 2)
		{
			for (std::size_t range = place_cells_number_range / 2; range > 1; range /= 2)
			{
	#pragma omp parallel for
				for (int k = 0; k < range; k++)
				{
					auto a = (k);
					auto b = (range + k);
					auto hypothesis_a = &hypothesis_map[a * hypothesis_map_stride * y_size];
					auto hypothesis_b = &hypothesis_map[b * hypothesis_map_stride * y_size];

					for (std::size_t row = 0; row < y_size; row++)
					{
						auto hypothesis_a_row = &hypothesis_a[row * hypothesis_map_stride];
						auto hypothesis_b_row = &hypothesis_b[row * hypothesis_map_stride];

						sum<Implementation>(hypothesis_a_row, hypothesis_b_row, x_size, hypothesis_a_row);
					}
				}
			}
			{
				auto hypothesis_a = &hypothesis_map[0];
				auto hypothesis_b = &hypothesis_map[hypothesis_map_stride * y_size];
	#pragma omp parallel for
				for (int row = 0; row < y_size; row++)
				{
					auto hypothesis_a_row = &hypothesis_a[row * hypothesis_map_stride];
					auto hypothesis_b_row = &hypothesis_b[row * hypothesis_map_stride];
					auto location_probability_row = &location_probability[row * location_probability_stride];

					sum<Implementation>(hypothesis_a_row, hypothesis_b_row, x_size, location_probability_row);
				}
			}
		}

		if (place_cells_number_remaining > 0)
		{
			auto hypothesis_k = &hypothesis_map[place_cells_number_range * hypothesis_map_stride * y_size];
			auto scale_k = set1_ps(scale[place_cells_number_range]);
	#pragma omp parallel for
			for (int row = 0; row < y_size; row++)
			{
				auto hypothesis_k_row = &hypothesis_k[row * hypothesis_map_stride];
				auto location_probability_row = &location_probability[row* location_probability_stride];

				weighted_acc<Implementation>(hypothesis_k_row, scale_k, x_size, location_probability_row);
			}
		}
		std::vector<float> max(y_size);
		std::vector<std::size_t> arg_max(y_size);

	#pragma omp parallel for
		for (int row = 0; row < y_size; row++)
		{
			const auto __y2 = set1_ps(y_grid_centered2[row]);
			auto location_probability_row = &location_probability[row* location_probability_stride];
			//circle<Implementation>(x_grid_centered2, x_size, __y2, __r2, location_probability_row);
			auto result = std::max_element(location_probability_row, location_probability_row + x_size);
			max[row] = *result;
			arg_max[row] = std::distance(location_probability_row, result);
		}

		auto result = std::max_element(max.begin(), max.end());
		row_decoded = std::distance(max.begin(), result);
		col_decoded = arg_max[row_decoded];

		*/
		//cv::Mat mat2(y_size, x_size, CV_32F, location_probability, location_probability_stride * sizeof(float));

		/*duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

		std::cout << "printf: " << duration << '\n';*/
}

template <TRN::CPU::Implementation Implementation>
static inline float  dot_product(const float *x, const float *a, const std::size_t &cols)
{
	std::size_t col = 0;
	auto __y = setzero_ps();
	/*if (cols - col > _32)
	{
		auto y0 = setzero_ps();
		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();
		auto y4 = setzero_ps();
		auto y5 = setzero_ps();
		auto y6 = setzero_ps();
		auto y7 = setzero_ps();
		auto y8 = setzero_ps();
		auto y9 = setzero_ps();
		auto y10 = setzero_ps();
		auto y11 = setzero_ps();
		auto y12 = setzero_ps();
		auto y13 = setzero_ps();
		auto y14 = setzero_ps();
		auto y15 = setzero_ps();
		auto y16 = setzero_ps();
		auto y17 = setzero_ps();
		auto y18 = setzero_ps();
		auto y19 = setzero_ps();
		auto y20 = setzero_ps();
		auto y21 = setzero_ps();
		auto y22 = setzero_ps();
		auto y23 = setzero_ps();
		auto y24 = setzero_ps();
		auto y25 = setzero_ps();
		auto y26 = setzero_ps();
		auto y27 = setzero_ps();
		auto y28 = setzero_ps();
		auto y29 = setzero_ps();
		auto y30 = setzero_ps();
		auto y31 = setzero_ps();
		for (; col + _32 - 1 < cols; col += _32)
		{
			y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
			y1 = mul_add_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), y1);
			y2 = mul_add_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2]), y2);
			y3 = mul_add_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3]), y3);
			y4 = mul_add_ps(load_ps(&a[col + _4]), load_ps(&x[col + _4]), y4);
			y5 = mul_add_ps(load_ps(&a[col + _5]), load_ps(&x[col + _5]), y5);
			y6 = mul_add_ps(load_ps(&a[col + _6]), load_ps(&x[col + _6]), y6);
			y7 = mul_add_ps(load_ps(&a[col + _7]), load_ps(&x[col + _7]), y7);
			y8 = mul_add_ps(load_ps(&a[col + _8]), load_ps(&x[col + _8]), y8);
			y9 = mul_add_ps(load_ps(&a[col + _9]), load_ps(&x[col + _9]), y9);
			y10 = mul_add_ps(load_ps(&a[col + _10]), load_ps(&x[col + _10]), y10);
			y11 = mul_add_ps(load_ps(&a[col + _11]), load_ps(&x[col + _11]), y11);
			y12 = mul_add_ps(load_ps(&a[col + _12]), load_ps(&x[col + _12]), y12);
			y13 = mul_add_ps(load_ps(&a[col + _13]), load_ps(&x[col + _13]), y13);
			y14 = mul_add_ps(load_ps(&a[col + _14]), load_ps(&x[col + _14]), y14);
			y15 = mul_add_ps(load_ps(&a[col + _15]), load_ps(&x[col + _15]), y15);
			y16 = mul_add_ps(load_ps(&a[col + _16]), load_ps(&x[col + _16]), y16);
			y17 = mul_add_ps(load_ps(&a[col + _17]), load_ps(&x[col + _17]), y17);
			y18 = mul_add_ps(load_ps(&a[col + _18]), load_ps(&x[col + _18]), y18);
			y19 = mul_add_ps(load_ps(&a[col + _19]), load_ps(&x[col + _19]), y19);
			y20 = mul_add_ps(load_ps(&a[col + _20]), load_ps(&x[col + _20]), y20);
			y21 = mul_add_ps(load_ps(&a[col + _21]), load_ps(&x[col + _21]), y21);
			y22 = mul_add_ps(load_ps(&a[col + _22]), load_ps(&x[col + _22]), y22);
			y23 = mul_add_ps(load_ps(&a[col + _23]), load_ps(&x[col + _23]), y23);
			y24 = mul_add_ps(load_ps(&a[col + _24]), load_ps(&x[col + _24]), y24);
			y25 = mul_add_ps(load_ps(&a[col + _25]), load_ps(&x[col + _25]), y25);
			y26 = mul_add_ps(load_ps(&a[col + _26]), load_ps(&x[col + _26]), y26);
			y27 = mul_add_ps(load_ps(&a[col + _27]), load_ps(&x[col + _27]), y27);
			y28 = mul_add_ps(load_ps(&a[col + _28]), load_ps(&x[col + _28]), y28);
			y29 = mul_add_ps(load_ps(&a[col + _29]), load_ps(&x[col + _29]), y29);
			y30 = mul_add_ps(load_ps(&a[col + _30]), load_ps(&x[col + _30]), y30);
			y31 = mul_add_ps(load_ps(&a[col + _31]), load_ps(&x[col + _31]), y31);
			
		}
		y0 = add_ps(y0, y1);
		y2 = add_ps(y2, y3);
		y4 = add_ps(y4, y5);
		y6 = add_ps(y6, y7);
		y0 = add_ps(y0, y9);
		y10 = add_ps(y10, y11);
		y12 = add_ps(y12, y13);
		y14 = add_ps(y14, y15);
		y16 = add_ps(y16, y17);
		y18 = add_ps(y18, y19);
		y20 = add_ps(y20, y21);
		y22 = add_ps(y22, y23);
		y24 = add_ps(y24, y25);
		y26 = add_ps(y26, y27);
		y28 = add_ps(y28, y29);
		y30 = add_ps(y30, y31);


		y0 = add_ps(y0, y2);
		y4 = add_ps(y4, y6);
		y8 = add_ps(y8, y10);
		y12 = add_ps(y12, y14);
		y16 = add_ps(y16, y18);
		y20 = add_ps(y20, y22);
		y24 = add_ps(y24, y16);
		y28 = add_ps(y28, y30);

		y0 = add_ps(y0, y4);
		y8 = add_ps(y8, y12);
		y16 = add_ps(y16, y20);
		y24 = add_ps(y24, y28);

		y0 = add_ps(y0, y8);
		y16 = add_ps(y16, y24);

		y0 = add_ps(y0, y16);
		__y = add_ps(__y, y0);
	}
	if (cols - col > 06)
	{
		auto y0 = setzero_ps();
		auto y1 = setzero_ps();
		auto y2 = setzero_ps();
		auto y3 = setzero_ps();
		auto y4 = setzero_ps();
		auto y5 = setzero_ps();
		auto y6 = setzero_ps();
		auto y7 = setzero_ps();
		auto y8 = setzero_ps();
		auto y9 = setzero_ps();
		auto y10 = setzero_ps();
		auto y11 = setzero_ps();
		auto y12 = setzero_ps();
		auto y13 = setzero_ps();
		auto y14 = setzero_ps();
		auto y15 = setzero_ps();
		for (; col + _16 - 1 < cols; col += _16)
		{
			y0= mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
			y1=mul_add_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), y1);
			y2=mul_add_ps(load_ps(&a[col + _2]), load_ps(&x[col + _2]), y2);
			y3=mul_add_ps(load_ps(&a[col + _3]), load_ps(&x[col + _3]), y3);
			y4=mul_add_ps(load_ps(&a[col + _4]), load_ps(&x[col + _4]), y4);
			y5=mul_add_ps(load_ps(&a[col + _5]), load_ps(&x[col + _5]), y5);
			y6=mul_add_ps(load_ps(&a[col + _6]), load_ps(&x[col + _6]), y6);
			y7=mul_add_ps(load_ps(&a[col + _7]), load_ps(&x[col + _7]), y7);
			y8=mul_add_ps(load_ps(&a[col + _8]), load_ps(&x[col + _8]), y8);
			y9=mul_add_ps(load_ps(&a[col + _9]), load_ps(&x[col + _9]), y9);
			y10=mul_add_ps(load_ps(&a[col + _10]), load_ps(&x[col + _10]), y10);
			y11=mul_add_ps(load_ps(&a[col + _11]), load_ps(&x[col + _11]), y11);
			y12=mul_add_ps(load_ps(&a[col + _12]), load_ps(&x[col + _12]), y12);
			y13=mul_add_ps(load_ps(&a[col + _13]), load_ps(&x[col + _13]), y13);
			y14=mul_add_ps(load_ps(&a[col + _14]), load_ps(&x[col + _14]), y14);
			y15=mul_add_ps(load_ps(&a[col + _15]), load_ps(&x[col + _15]), y15);
		}
		y0 = add_ps(y0, y1);
		y2 = add_ps(y2, y3);
		y4 = add_ps(y4, y5);
		y6 = add_ps(y6, y7);
		y0 = add_ps(y0, y9);
		y10 = add_ps(y10, y11);
		y12 = add_ps(y12, y13);
		y14 = add_ps(y14, y15);

		y0 = add_ps(y0, y2);
		y4 = add_ps(y4, y6);
		y8 = add_ps(y8, y10);
		y12 = add_ps(y12, y14);

		y0 = add_ps(y0, y4);
		y8 = add_ps(y8, y12);

		y0 = add_ps(y0, y8);
		__y = add_ps(__y, y0);
	}
	*/if (cols - col > _8)
	{
		auto y0 = setzero_ps();
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
		__y = add_ps(__y, y0);
	}
	if (cols - col > _4)
	{
		auto y0 = setzero_ps();
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
		__y = add_ps(__y, y0);
	}
	if (cols - col > _2)
	{
		auto y0 = setzero_ps();
		auto y1 = setzero_ps();
		for (; col + _2 - 1 < cols; col += _2)
		{
		   y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
		   y1 = mul_add_ps(load_ps(&a[col + _1]), load_ps(&x[col + _1]), y1);
		}
		y0 = add_ps(y0, y1);
		__y = add_ps(__y, y0);
	}
	 if (cols - col > 0)
	{
		auto y0 = setzero_ps();
		for (; col  < cols; col += _1)
		{
			y0 = mul_add_ps(load_ps(&a[col + _0]), load_ps(&x[col + _0]), y0);
		}
		__y = add_ps(__y, y0);
	}
	
	return hsum_ps(__y);
}


/*template <TRN::CPU::Implementation Implementation, const std::size_t unroll_factor>
static inline std::size_t matrix_vector_product_loop(const float *x, const float *A, const std::size_t &rows, const std::size_t &cols, const std::size_t &stride, float *y)
{
	const std::size_t chunk_stride = TRN::CPU::Traits<Implementation>::step * unroll_factor;
	const std::size_t rows_remaining = rows % chunk_stride;
	const std::size_t rows_range = rows - rows_remaining;
	const std::size_t chunks = rows / chunk_stride;

#pragma omp parallel for
	for (int chunk = 0; chunk < chunks; chunk++)
	{
		const std::size_t row = chunk * chunk_stride;
		for (std::size_t offset = 0; offset < chunk_stride; offset += TRN::CPU::Traits<Implementation>::step)
		{
			typename TRN::CPU::Traits<Implementation>::type __y;
			for (std::size_t s = 0; s < TRN::CPU::Traits<Implementation>::step; s++)
			{
				set_element(dot_product<Implementation>(x, &A[stride * (row + offset + s)], stride), s, __y);
			}
			stream_ps(&y[row + offset], __y);
		}
	}
	return rows_range;
}
*/
template <TRN::CPU::Implementation Implementation>
static inline void matrix_vector_product(const float *x, const float *A, const std::size_t &a_rows, const std::size_t &a_cols, const std::size_t &a_stride, float *y)
{
	//std::size_t rows = 0;

	/*if (a_rows - rows > 16 * TRN::CPU::Traits<Implementation>::step * omp_get_max_threads())
	{
		rows += matrix_vector_product_loop<Implementation, 16>(x, &A[rows * a_stride], a_rows - rows, a_cols, a_stride, &y[rows]);
	}
	if (a_rows - rows > 8 * TRN::CPU::Traits<Implementation>::step * omp_get_max_threads())
	{
		rows += matrix_vector_product_loop<Implementation, 8>(x, &A[rows * a_stride], a_rows - rows, a_cols, a_stride, &y[rows]);
	}*/
	/*if (a_rows - rows > 4 * TRN::CPU::Traits<Implementation>::step* omp_get_max_threads())
	{
		rows += matrix_vector_product_loop<Implementation, 4>(x, &A[rows * a_stride], a_rows - rows, a_cols, a_stride, &y[rows]);
	}
	if (a_rows - rows > 2 * TRN::CPU::Traits<Implementation>::step* omp_get_max_threads() )
	{
		rows += matrix_vector_product_loop<Implementation, 2>(x, &A[rows * a_stride], a_rows - rows, a_cols, a_stride, &y[rows]);
	}*/
	/*if (a_rows - rows > 0)
	{
		rows += matrix_vector_product_loop<Implementation, 1>(x, &A[rows * a_stride], a_rows - rows, a_cols, a_stride, &y[rows]);
	}*/

#pragma omp parallel for
	for (int row = 0; row < a_rows; row++)
	{
		y[row] = dot_product<Implementation>(x, &A[a_stride * row], a_cols);
	}
}
/*template <TRN::CPU::Implementation Implementation, const std::size_t unroll_factor>
static inline std::size_t update_reservoir_loop(const float * const w_in, const std::size_t &w_in_rows, const std::size_t &w_in_cols, const  std::size_t &w_in_stride,
	const float * const x_in,
	const float * const u_ffwd,
	const typename TRN::CPU::Traits<Implementation>::type &leak_rate,
	float * p,
	float *  x_res
)
{
	const std::size_t chunk_stride = TRN::CPU::Traits<Implementation>::step * unroll_factor;
	const std::size_t rows_remaining = w_in_rows % chunk_stride;
	const std::size_t rows_range = w_in_rows - rows_remaining;
	const std::size_t chunks = w_in_rows / chunk_stride;

#pragma omp parallel for
	for (int chunk = 0; chunk < chunks; chunk++)
	{
		const std::size_t row = chunk * chunk_stride;

		for (int offset = 0; offset < chunk_stride; offset += TRN::CPU::Traits<Implementation>::step)
		{
			auto __u = load_ps(&u_ffwd[row + offset]);
			auto __p = load_ps(&p[row + offset]);
			typename TRN::CPU::Traits<Implementation>::type __c;
			for (int s = 0; s < TRN::CPU::Traits<Implementation>::step; s++)
			{
				set_element(dot_product<Implementation>(x_in, &w_in[w_in_stride * (row + offset + s)], w_in_stride), s, __c);
			}
			__p = mul_add_ps(leak_rate, sub_ps(add_ps(__u, __c), __p), __p);
			stream_ps(&x_res[row + offset], tanh_ps(__p));
			stream_ps(&p[row + offset], __p);
		}

	}
	return rows_range;
}*/

template <TRN::CPU::Implementation Implementation>
static inline void update_reservoir(const float * const w_in, const std::size_t &w_in_rows, const std::size_t &w_in_cols, const  std::size_t &w_in_stride,
	const float * const x_in,
	const float * const u_ffwd,
	const typename TRN::CPU::Traits<Implementation>::type &leak_rate,
	float * p,
	float *  x_res
)
{
	const std::size_t chunk_stride = TRN::CPU::Traits<Implementation>::step;
	const std::size_t rows_remaining = w_in_rows % chunk_stride;
	const std::size_t rows_range = w_in_rows - rows_remaining;
	const std::size_t chunks = w_in_rows / chunk_stride;
#pragma omp parallel for
	for (int chunk = 0; chunk < chunks; chunk++)
	{
		const std::size_t row = chunk * chunk_stride;

		auto __u = load_ps(&u_ffwd[row]);
		auto __p = load_ps(&p[row]);
		typename TRN::CPU::Traits<Implementation>::type __c;
		for (int s = 0; s < TRN::CPU::Traits<Implementation>::step; s++)
		{
			set_element(dot_product<Implementation>(x_in, &w_in[w_in_stride * (row + s)], w_in_stride), s, __c);
		}

		__p = mul_add_ps(leak_rate, sub_ps(add_ps(__u, __c), __p), __p);
		stream_ps(&x_res[row], tanh_ps(__p));
		stream_ps(&p[row], __p);
	}
}


template <TRN::CPU::Implementation Implementation>
static inline void premultiply(
	float *u_ffwd, const std::size_t &u_ffwd_rows, const std::size_t &u_ffwd_cols, const std::size_t &u_ffwd_stride,
	const float *w_ffwd, const std::size_t &w_ffwd_rows, const std::size_t &w_ffwd_cols, const std::size_t &w_ffwd_stride,
	const float *incoming_samples, const std::size_t &incoming_rows, const std::size_t &incoming_cols, const std::size_t &incoming_stride)
{
#pragma omp parallel for 
	for (int t = 0; t < incoming_rows; t++)
	{
		auto incoming = &incoming_samples[t * incoming_stride];
		float *u = &u_ffwd[t * u_ffwd_stride];
		matrix_vector_product<Implementation>(incoming, w_ffwd, w_ffwd_rows, w_ffwd_cols, w_ffwd_stride, u);
	}

}

template<TRN::CPU::Implementation Implementation, LearningRule rule>
struct learn_weights
{
	inline void operator () (float *w, const std::size_t &w_rows, const std::size_t &w_cols, const std::size_t &w_stride,
		const float *x_pre, const std::size_t &x_pre_rows, const std::size_t &x_pre_cols, const std::size_t &x_pre_stride,
		float *x_post, const std::size_t &x_post_rows, const std::size_t &x_post_cols, const std::size_t &x_post_stride,
		const float *d_post, const std::size_t &d_post_rows, const std::size_t &d_post_cols, const std::size_t &d_post_stride)
	{
	}
};

template<TRN::CPU::Implementation Implementation>
struct learn_weights<Implementation, LearningRule::NOTHING>
{
	inline void operator () (float *w, const std::size_t &w_rows, const std::size_t &w_cols, const std::size_t &w_stride,
		const float *x_pre, const std::size_t &x_pre_rows, const std::size_t &x_pre_cols, const std::size_t &x_pre_stride,
		float *x_post, const std::size_t &x_post_rows, const std::size_t &x_post_cols, const std::size_t &x_post_stride,
		const float *d_post, const std::size_t &d_post_rows, const std::size_t &d_post_cols, const std::size_t &d_post_stride)
	{

	}
};


template<TRN::CPU::Implementation Implementation>
struct learn_weights<Implementation, LearningRule::WIDROW_HOFF>
{
private:
	const typename TRN::CPU::Traits<Implementation>::type learning_rate;
public:
	learn_weights(const float &learning_rate) :  learning_rate(set1_ps(learning_rate))
	{

	}
	~learn_weights()
	{
	
	}


private :
	template<const std::size_t unroll_factor>
	std::size_t  learn_loop(float *w, const std::size_t &w_rows, const std::size_t &w_cols, const std::size_t &w_stride,
		const float *x_pre, const std::size_t &x_pre_rows, const std::size_t &x_pre_cols, const std::size_t &x_pre_stride,
		float *x_post, const std::size_t &x_post_rows, const std::size_t &x_post_cols, const std::size_t &x_post_stride,
		const float *d_post, const std::size_t &d_post_rows, const std::size_t &d_post_cols, const std::size_t &d_post_stride)
	{
		const std::size_t chunk_stride = TRN::CPU::Traits<Implementation>::step * unroll_factor;
		const std::size_t rows_remaining = w_rows % chunk_stride;
		const std::size_t rows_range = w_rows - rows_remaining;
		const std::size_t chunks = w_rows / chunk_stride;

#pragma omp parallel for
		for (int chunk = 0; chunk < chunks; chunk++)
		{
			const std::size_t row = chunk * chunk_stride;
			for (int offset = 0; offset < chunk_stride; offset += TRN::CPU::Traits<Implementation>::step)
			{
				const auto __e_row = mul_ps(learning_rate, sub_ps(load_ps(&d_post[row + offset]), load_ps(&x_post[row + offset])));
				for (int s = 0; s < TRN::CPU::Traits<Implementation>::step; s++)
				{
					const auto __e = set1_ps(get_element(s, __e_row));

					float *w_row = &w[(row + offset + s)* w_stride];
					std::size_t col = 0;
					/*if (w_cols - col > 06)
					{
						for (; col + _16 - 1 < w_cols; col += _16)
						{
							stream_ps(&w_row[col + _0], mul_add_ps(__e, load_ps(&x_pre[col + _0]), load_ps(&w_row[col + _0])));
							stream_ps(&w_row[col + _1], mul_add_ps(__e, load_ps(&x_pre[col + _1]), load_ps(&w_row[col + _1])));
							stream_ps(&w_row[col + _2], mul_add_ps(__e, load_ps(&x_pre[col + _2]), load_ps(&w_row[col + _2])));
							stream_ps(&w_row[col + _3], mul_add_ps(__e, load_ps(&x_pre[col + _3]), load_ps(&w_row[col + _3])));
							stream_ps(&w_row[col + _4], mul_add_ps(__e, load_ps(&x_pre[col + _4]), load_ps(&w_row[col + _4])));
							stream_ps(&w_row[col + _5], mul_add_ps(__e, load_ps(&x_pre[col + _5]), load_ps(&w_row[col + _5])));
							stream_ps(&w_row[col + _6], mul_add_ps(__e, load_ps(&x_pre[col + _6]), load_ps(&w_row[col + _6])));
							stream_ps(&w_row[col + _7], mul_add_ps(__e, load_ps(&x_pre[col + _7]), load_ps(&w_row[col + _7])));
							stream_ps(&w_row[col + _8], mul_add_ps(__e, load_ps(&x_pre[col + _8]), load_ps(&w_row[col + _8])));
							stream_ps(&w_row[col + _9], mul_add_ps(__e, load_ps(&x_pre[col + _9]), load_ps(&w_row[col + _9])));
							stream_ps(&w_row[col + _10], mul_add_ps(__e, load_ps(&x_pre[col + _10]), load_ps(&w_row[col + _10])));
							stream_ps(&w_row[col + _11], mul_add_ps(__e, load_ps(&x_pre[col + _11]), load_ps(&w_row[col + _11])));
							stream_ps(&w_row[col + _12], mul_add_ps(__e, load_ps(&x_pre[col + _12]), load_ps(&w_row[col + _12])));
							stream_ps(&w_row[col + _13], mul_add_ps(__e, load_ps(&x_pre[col + _13]), load_ps(&w_row[col + _13])));
							stream_ps(&w_row[col + _14], mul_add_ps(__e, load_ps(&x_pre[col + _14]), load_ps(&w_row[col + _14])));
							stream_ps(&w_row[col + _15], mul_add_ps(__e, load_ps(&x_pre[col + _15]), load_ps(&w_row[col + _15])));
						}
					}*/
					if (w_cols - col > _8)
					{
						for (; col + _8 - 1 < w_cols; col += _8)
						{
							stream_ps(&w_row[col + _0], mul_add_ps(__e, load_ps(&x_pre[col + _0]), load_ps(&w_row[col + _0])));
							stream_ps(&w_row[col + _1], mul_add_ps(__e, load_ps(&x_pre[col + _1]), load_ps(&w_row[col + _1])));
							stream_ps(&w_row[col + _2], mul_add_ps(__e, load_ps(&x_pre[col + _2]), load_ps(&w_row[col + _2])));
							stream_ps(&w_row[col + _3], mul_add_ps(__e, load_ps(&x_pre[col + _3]), load_ps(&w_row[col + _3])));
							stream_ps(&w_row[col + _4], mul_add_ps(__e, load_ps(&x_pre[col + _4]), load_ps(&w_row[col + _4])));
							stream_ps(&w_row[col + _5], mul_add_ps(__e, load_ps(&x_pre[col + _5]), load_ps(&w_row[col + _5])));
							stream_ps(&w_row[col + _6], mul_add_ps(__e, load_ps(&x_pre[col + _6]), load_ps(&w_row[col + _6])));
							stream_ps(&w_row[col + _7], mul_add_ps(__e, load_ps(&x_pre[col + _7]), load_ps(&w_row[col + _7])));
						}
					}
					if (w_cols - col > _4)
					{
						for (; col + _4 - 1 < w_cols; col += _4)
						{
							stream_ps(&w_row[col + _0], mul_add_ps(__e, load_ps(&x_pre[col + _0]), load_ps(&w_row[col + _0])));
							stream_ps(&w_row[col + _1], mul_add_ps(__e, load_ps(&x_pre[col + _1]), load_ps(&w_row[col + _1])));
							stream_ps(&w_row[col + _2], mul_add_ps(__e, load_ps(&x_pre[col + _2]), load_ps(&w_row[col + _2])));
							stream_ps(&w_row[col + _3], mul_add_ps(__e, load_ps(&x_pre[col + _3]), load_ps(&w_row[col + _3])));
						}
					}
					if (w_cols - col > _2)
					{
						for (; col + _2 - 1 < w_cols; col += _2)
						{
							stream_ps(&w_row[col + _0], mul_add_ps(__e, load_ps(&x_pre[col + _0]), load_ps(&w_row[col + _0])));
							stream_ps(&w_row[col + _1], mul_add_ps(__e, load_ps(&x_pre[col + _1]), load_ps(&w_row[col + _1])));
						}
					}
					if (w_cols - col > 0)
					{
						for (; col  < w_cols; col += _1)
						{
							stream_ps(&w_row[col + _0], mul_add_ps(__e, load_ps(&x_pre[col + _0]), load_ps(&w_row[col + _0])));
						}
					}
				}
			}
		}
		return rows_range;
	}

	public :
	inline void operator () (float *w, const std::size_t &w_rows, const std::size_t &w_cols, const std::size_t &w_stride,
		const float *x_pre, const std::size_t &x_pre_rows, const std::size_t &x_pre_cols, const std::size_t &x_pre_stride,
		float *x_post, const std::size_t &x_post_rows, const std::size_t &x_post_cols, const std::size_t &x_post_stride,
		const float *d_post, const std::size_t &d_post_rows, const std::size_t &d_post_cols, const std::size_t &d_post_stride)
	{
		//cv::Mat u(w_rows, w_cols, CV_32F, (void *)w, w_stride * sizeof(float));

/*#pragma omp parallel for
		for (int row = 0; row < w_rows; row++)
		{
			auto w_row = &w[row * w_stride];

	
			const float err = (d_post[row] - x_post[row]) * lr;
			for (std::size_t col = 0; col < w_cols; col++)
			{
				w_row[col] += err * x_pre[col];
			}
		}*/
		std::size_t rows = 0;
		/*if (w_rows - rows > 16 * TRN::CPU::Traits<Implementation>::step * omp_get_max_threads())
		{
			rows += learn_loop<16>(&w[rows * w_stride], w_rows -rows, w_cols, w_stride, &x_pre[rows * w_stride], x_pre_rows - rows, x_pre_cols, x_pre_stride, x_post, x_post_rows, x_post_cols, x_post_stride, d_post, d_post_rows, d_post_cols, d_post_stride);
		}
		if (w_rows - rows > 8 * TRN::CPU::Traits<Implementation>::step * omp_get_max_threads())
		{
			rows += learn_loop<8>(&w[rows * w_stride], w_rows - rows, w_cols, w_stride, &x_pre[rows * w_stride], x_pre_rows - rows, x_pre_cols, x_pre_stride, x_post, x_post_rows, x_post_cols, x_post_stride, d_post, d_post_rows, d_post_cols, d_post_stride);
		}
		if (w_rows - rows > 4 * TRN::CPU::Traits<Implementation>::step * omp_get_max_threads())
		{
			rows += learn_loop<4>(&w[rows * w_stride], w_rows - rows, w_cols, w_stride, &x_pre[rows * w_stride], x_pre_rows - rows, x_pre_cols, x_pre_stride, x_post, x_post_rows, x_post_cols, x_post_stride, d_post, d_post_rows, d_post_cols, d_post_stride);
		}
		if (w_rows - rows > 2 * TRN::CPU::Traits<Implementation>::step * omp_get_max_threads())
		{
			rows += learn_loop<2>(&w[rows * w_stride], w_rows - rows, w_cols, w_stride, &x_pre[rows * w_stride], x_pre_rows - rows, x_pre_cols, x_pre_stride, x_post, x_post_rows, x_post_cols, x_post_stride, d_post, d_post_rows, d_post_cols, d_post_stride);
		}*/
		if (w_rows - rows > 0)
		{
			rows += learn_loop<1>(&w[rows * w_stride], w_rows - rows, w_cols, w_stride, &x_pre[rows * w_stride], x_pre_rows - rows, x_pre_cols, x_pre_stride, x_post, x_post_rows, x_post_cols, x_post_stride, d_post, d_post_rows, d_post_cols, d_post_stride);
		}
	}
};

template<TRN::CPU::Implementation Implementation, LearningRule rule>
static inline void update_readout(

	float *w, const std::size_t &w_rows, const std::size_t &w_cols, const std::size_t &w_stride,
	const float *x_pre, const std::size_t &x_pre_rows, const std::size_t &x_pre_cols, const std::size_t &x_pre_stride,
	float *x_post, const std::size_t &x_post_rows, const std::size_t &x_post_cols, const std::size_t &x_post_stride,
	const float *d_post, const std::size_t &d_post_rows, const std::size_t &d_post_cols, const std::size_t &d_post_stride,
	learn_weights<Implementation, rule> &learn)
{
	matrix_vector_product<Implementation>(x_pre, w, w_rows, w_cols, w_stride, x_post);
	learn(w, w_rows, w_cols, w_stride, x_pre, x_pre_rows, x_pre_cols, x_pre_stride, x_post, x_post_rows, x_post_cols, x_post_stride, d_post, d_post_rows, d_post_cols, d_post_stride);
}





template<TRN::CPU::Implementation Implementation, bool gather_states>
struct copy_states
{
	void operator() (const float *const src, const std::size_t &src_stride, float *dst, const std::size_t &dst_stride, const std::size_t &size)
	{

	}
};
template<TRN::CPU::Implementation Implementation>
struct copy_states<Implementation, true>
{
	void operator () (const float *const src, const std::size_t &src_stride, float *dst, const std::size_t &dst_stride, const std::size_t &size)
	{
		TRN::CPU::Memory<Implementation>::copy_implementation(src, dst, sizeof(float), size, 1, src_stride, dst_stride, true);
	}
};

template <TRN::CPU::Implementation Implementation, bool overwrite_states>
struct initialize_states
{
	void operator () (unsigned long &seed, float *ptr, const std::size_t &rows, const std::size_t &cols, const std::size_t &stride, const float &initial_state_scale)
	{

	}
};

template <TRN::CPU::Implementation Implementation>
struct initialize_states<Implementation, true>
{
	void operator () (unsigned long &seed, float *ptr, const std::size_t &rows, const std::size_t &cols, const std::size_t &stride, const float &initial_state_scale)
	{
		
		TRN::CPU::Random::uniform_implementation(ptr, seed, rows, cols, stride, -initial_state_scale, initial_state_scale, 0.0f);
		seed += rows * cols;
	}
};

template<TRN::CPU::Implementation Implementation, bool gather_states, bool overwrite_states, LearningRule rule>
static inline void update_model(
	unsigned long &seed,
	const std::size_t &stimulus_stride, const std::size_t &reservoir_stride, const std::size_t &prediction_stride,
	const std::size_t &stimulus_size, const std::size_t &reservoir_size, const std::size_t &prediction_size,
	const float &leak_rate, const float &initial_state_scale,
	float *p, const std::size_t &p_rows, const std::size_t &p_cols, const std::size_t &p_stride,float *x_in, const std::size_t &x_in_rows, const std::size_t &x_in_cols, const std::size_t &x_in_stride,
	const float *w_ffwd, const std::size_t &w_ffwd_rows, const std::size_t &w_ffwd_cols, const std::size_t &w_ffwd_stride,
	const float *x_in, const std::size_t &x_in_rows, const std::size_t &x_in_cols, const std::size_t &x_in_stride,
	const float *w_in, const std::size_t &w_in_rows, const std::size_t &w_in_cols, const std::size_t &w_in_stride,
	float *x_res, const std::size_t &x_res_rows, const std::size_t &x_res_cols, const std::size_t &x_res_stride,
	float *x_ro, const std::size_t &x_ro_rows, const std::size_t &x_ro_cols, const std::size_t &x_ro_stride,
	float *w_ro, const std::size_t &w_ro_rows, const std::size_t &w_ro_cols, const std::size_t &w_ro_stride,

	const float *incoming_samples, const std::size_t &incoming_rows, const std::size_t &incoming_cols, const std::size_t &incoming_stride,
	const float *expected_samples, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,

	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,
	learn_weights<Implementation, rule> &learn)
{
	float *c;
	std::size_t c_stride;

	float *u_ffwd;
	std::size_t u_ffwd_stride;
	const std::size_t u_ffwd_rows = incoming_rows;
	const std::size_t u_ffwd_cols = x_res_cols;

	auto __leak_rate = set1_ps(leak_rate);


	static initialize_states<Implementation, overwrite_states> initializer;
	static copy_states<Implementation, gather_states> copier;

	TRN::CPU::Memory<Implementation>::allocate_implementation((void **)&u_ffwd, u_ffwd_stride, sizeof(float), u_ffwd_cols, u_ffwd_rows);

	premultiply<Implementation>(
		u_ffwd, u_ffwd_rows, u_ffwd_cols, u_ffwd_stride,
		w_ffwd, w_ffwd_rows, w_ffwd_cols, w_ffwd_stride,
		incoming_samples, incoming_rows, incoming_cols, incoming_stride);
	
	int ts = 0;
	for (int r = 0; r < repetitions; r++)
	{
		initializer(seed, p, 1, p_cols, p_stride, initial_state_scale);
		initializer(seed, (float *)x_in, x_in_rows, x_in_cols, x_in_stride, initial_state_scale);


		auto t0 = offsets[r];
		auto tn = t0 + durations[r];

		for (int t = t0; t < tn; t++, ts++)
		{
	
			update_reservoir<Implementation>(w_in, w_in_rows, w_in_cols, w_in_stride, x_in, &u_ffwd[t * u_ffwd_stride], __leak_rate, p, x_res);
			update_readout<Implementation, rule> (w_ro, w_ro_rows, w_ro_cols, w_ro_stride,
			x_res, x_res_rows, x_res_cols, x_res_stride,
			x_ro, x_ro_rows, x_ro_cols, x_ro_stride,
			&expected_samples[t * expected_stride], 1, expected_cols, expected_stride, learn);

			//std::cout << d << std::endl;

			copier(&incoming_samples[t * incoming_stride], incoming_stride, &states_samples[ts*states_stride], states_stride, incoming_cols);
			copier(x_res, x_res_stride, &states_samples[ts*states_stride] + stimulus_stride, states_stride, x_res_cols);
			copier(x_ro, x_ro_stride, &states_samples[ts*states_stride] + stimulus_stride + reservoir_stride, states_stride, x_ro_cols);
			copier(&expected_samples[t * expected_stride], expected_stride, &states_samples[ts*states_stride] + stimulus_stride + reservoir_stride + prediction_stride, states_stride, expected_cols);
		}
	}

	TRN::CPU::Memory<Implementation>::deallocate_implementation(u_ffwd);
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
	float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_stride,
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride,

	const float &learning_rate)
{
	/*if (states_samples == NULL)
	{
		update_model<Implementation, false, true, LearningRule::WIDROW_HOFF>(
			seed, 
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size, leak_rate, initial_state_scale,
			p, p_rows, p_cols, p_stride,
			w_ffwd, w_ffwd_rows, w_ffwd_cols, w_ffwd_stride,
			x_in, x_in_rows, x_in_cols, x_in_stride,
			w_in, w_in_rows, w_in_cols, w_in_stride,
			x_res, x_res_rows, x_res_cols, x_res_stride,
			x_ro, x_ro_rows, x_ro_cols, x_ro_stride,
			w_ro, w_ro_rows, w_ro_cols, w_ro_stride,
			incoming_samples, incoming_rows, incoming_cols, incoming_stride,
			expected_samples, expected_rows, expected_cols, expected_stride,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride,
			learn_weights<Implementation, LearningRule::WIDROW_HOFF>(learning_rate)
			);
	}
	else
	{
		update_model<Implementation, true, true, LearningRule::WIDROW_HOFF>(
			seed, 
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size, leak_rate, initial_state_scale,
			p, p_rows, p_cols, p_stride,
			w_ffwd, w_ffwd_rows, w_ffwd_cols, w_ffwd_stride,
			x_in, x_in_rows, x_in_cols, x_in_stride,
			w_in, w_in_rows, w_in_cols, w_in_stride,
			x_res, x_res_rows, x_res_cols, x_res_stride,
			x_ro, x_ro_rows, x_ro_cols, x_ro_stride,
			w_ro, w_ro_rows, w_ro_cols, w_ro_stride,
			incoming_samples, incoming_rows, incoming_cols, incoming_stride,
			expected_samples, expected_rows, expected_cols, expected_stride,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride,
			learn_weights<Implementation, LearningRule::WIDROW_HOFF>(learning_rate)
			);
	}*/
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
	float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_stride,
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{


	/*if (states_samples == NULL)
	{
		update_model<Implementation, false, true, LearningRule::NOTHING>(
			seed,
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size, leak_rate, initial_state_scale,
			p, p_rows, p_cols, p_stride,
			w_ffwd, w_ffwd_rows, w_ffwd_cols, w_ffwd_stride,
			x_in, x_in_rows, x_in_cols, x_in_stride,
			w_in, w_in_rows, w_in_cols, w_in_stride,
			x_res, x_res_rows, x_res_cols, x_res_stride,
			x_ro, x_ro_rows, x_ro_cols, x_ro_stride,
			w_ro, w_ro_rows, w_ro_cols, w_ro_stride,
			incoming_samples, incoming_rows, incoming_cols, incoming_stride,
			expected_samples, expected_rows, expected_cols, expected_stride,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride,
			learn_weights<Implementation, LearningRule::NOTHING>()
			);
	}
	else
	{
		update_model<Implementation, true, true, LearningRule::NOTHING>(
			seed,
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size, leak_rate, initial_state_scale,
			p, p_rows, p_cols, p_stride,
			w_ffwd, w_ffwd_rows, w_ffwd_cols, w_ffwd_stride,
			x_in, x_in_rows, x_in_cols, x_in_stride,
			w_in, w_in_rows, w_in_cols, w_in_stride,
			x_res, x_res_rows, x_res_cols, x_res_stride,
			x_ro, x_ro_rows, x_ro_cols, x_ro_stride,
			w_ro, w_ro_rows, w_ro_cols, w_ro_stride,
			incoming_samples, incoming_rows, incoming_cols, incoming_stride,
			expected_samples, expected_rows, expected_cols, expected_stride,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride,
			learn_weights<Implementation, LearningRule::NOTHING>()
			);
	}*/

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
	float **batched_error, const std::size_t *batched_error_rows, const std::size_t *batched_error_cols, const std::size_t *batched_error_stride,
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{
	//std::unique_lock<std::mutex> lock(mutex);
	/*if (states_samples == NULL)
	{
		update_model<Implementation, false, false, LearningRule::NOTHING>(
			seed,
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size, leak_rate, initial_state_scale,
			p, p_rows, p_cols, p_stride,
			w_ffwd, w_ffwd_rows, w_ffwd_cols, w_ffwd_stride,
			x_in, x_in_rows, x_in_cols, x_in_stride,
			w_in, w_in_rows, w_in_cols, w_in_stride,
			x_res, x_res_rows, x_res_cols, x_res_stride,
			x_ro, x_ro_rows, x_ro_cols, x_ro_stride,
			w_ro, w_ro_rows, w_ro_cols, w_ro_stride,
			incoming_samples, incoming_rows, incoming_cols, incoming_stride,
			expected_samples, expected_rows, expected_cols, expected_stride,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride,
			learn_weights<Implementation, LearningRule::NOTHING>()
			);
	}
	else
	{
		update_model<Implementation, true, false, LearningRule::NOTHING>(
			seed,
			stimulus_stride, reservoir_stride, prediction_stride,
			stimulus_size, reservoir_size, prediction_size, leak_rate, initial_state_scale,
			p, p_rows, p_cols, p_stride,
			w_ffwd, w_ffwd_rows, w_ffwd_cols, w_ffwd_stride,
			x_in, x_in_rows, x_in_cols, x_in_stride,
			w_in, w_in_rows, w_in_cols, w_in_stride,
			x_res, x_res_rows, x_res_cols, x_res_stride,
			x_ro, x_ro_rows, x_ro_cols, x_ro_stride,
			w_ro, w_ro_rows, w_ro_cols, w_ro_stride,
			incoming_samples, incoming_rows, incoming_cols, incoming_stride,
			expected_samples, expected_rows, expected_cols, expected_stride,
			offsets, durations, repetitions,
			states_samples, states_rows, states_cols, states_stride,
			learn_weights<Implementation, LearningRule::NOTHING>()
			);
	}*/

}


template <TRN::CPU::Implementation Implementation>
std::shared_ptr<TRN::CPU::Algorithm<Implementation>> TRN::CPU::Algorithm<Implementation>::create()
{
	return std::make_shared<TRN::CPU::Algorithm<Implementation>>();
}





