#include "stdafx.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include "Algorithm.cuh"
#include "Random.cuh"
#include "Driver.cuh"

#include <helper_math.h>
#include <cub/cub.cuh>

#define BLOCK_X 32
#define BLOCK_Y 32
#define warpSize 32

/*template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
			file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
		DEVICE_RESET
			// Make sure we call CUDA Device Reset before exiting
			exit(EXIT_FAILURE);
	}
}*/


// This will output the proper CUDA error strings in the event that a CUDA host call returns an error


//#define checkCudaErrors(val)           throw_on_cuda_error ( (val), #val, __FILE__, __LINE__ )

__inline__ __device__
static bool isnan(float4 v)
{
	return isnan(v.x) || isnan(v.y) || isnan(v.z) || isnan(v.w);
}

#define warpSize 32
__inline__ __device__
static float warpReduceSum(float val)
{
	val += __shfl_down(val, 16);
	val += __shfl_down(val, 8);
	val += __shfl_down(val, 4);
	val += __shfl_down(val, 2);
	val += __shfl_down(val, 1);

	return val;
}
static inline __device__
float4 tanhf(const float4 &a)
{
	float4 b;

	b.x = tanhf(a.x);
	b.y = tanhf(a.y);
	b.z = tanhf(a.z);
	b.w = tanhf(a.w);

	return b;
}
static inline __device__
float4 expf(const float4 &a)
{
	float4 b;

	b.x = expf(a.x);
	b.y = expf(a.y);
	b.z = expf(a.z);
	b.w = expf(a.w);

	return b;
}
__global__
static void scale_kernel(float ** __restrict__ a,
	const int batch_size, const int rows, const int cols, const int stride, const float scale)

{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int col = blockIdx.y * blockDim.y + threadIdx.y; col < cols; col += gridDim.y * blockDim.y)
		{
			for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows; row += gridDim.x * blockDim.x)
			{
				a[batch][col * stride + row] *= scale;
			}
		}
	}
}


__global__
static void mean_square_error_naive_kernel(
	const int batch_size, const int rows, const int cols, const float scale,
	const float ** __restrict__ batched_predicted, const int batched_predicted_stride, 
	const float * __restrict__  expected,const int expected_stride,
	float *__restrict__  result, const int result_stride

)
{
	
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		const float *predicted = batched_predicted[batch];
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rows; row += gridDim.y * blockDim.y)
		{
			float sum = 0.0f;
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x)
			{
				float d = predicted[row * batched_predicted_stride + col] - expected[row * expected_stride + col];

				sum += (d * d);
			}

			sum = warpReduceSum(sum);
			if ((threadIdx.x & 31) == 0)
				atomicAdd(&result[row * result_stride + batch], sum * scale);
		}
	}
}

__global__
static void mean_square_error_float4_kernel(
	const int batch_size, const int rows, const int cols, const float scale,
	const float ** __restrict__ batched_predicted, const int batched_predicted_stride,
	const float * __restrict__  expected, const int expected_stride,
	float *__restrict__  result, const int result_stride

)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		const float *predicted = batched_predicted[batch];
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rows; row += gridDim.y * blockDim.y)
		{
			float sum = 0.0f;

			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x)
			{
				 float4 p = reinterpret_cast<float4 *>(const_cast<float *>(&predicted[row * batched_predicted_stride]))[col];
				 float4 e = reinterpret_cast<float4 *>(const_cast<float *>(&expected[row * expected_stride]))[col];
				 float4 d = p - e;
				 float4 c = (d * d);
				sum += c.x + c.y + c.z + c.w;
			}

			sum = warpReduceSum(sum);
			if ((threadIdx.x & 31) == 0)
				atomicAdd(&result[row * result_stride + batch], sum * scale);
		}
	}
}

void compute_mean_square_error
(
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const std::size_t &batch_size,
	const float **batched_predicted, const std::size_t &batched_predicted_rows, const std::size_t &batched_predicted_cols, const std::size_t &batched_predicted_stride,
	const float *expected, const std::size_t &expected_rows, const std::size_t &expected_cols, const std::size_t &expected_stride,
	float *result, const std::size_t &result_rows, const std::size_t &result_cols, const std::size_t &result_stride
)
{
	assert(batched_predicted_rows == expected_rows);
	assert(expected_rows == result_rows);
	assert(batched_predicted_cols == expected_cols);
	assert(result_cols == batch_size);
	const float scale = 1.0f / expected_cols;
	checkCudaErrors(cudaMemset2DAsync(result, result_stride * sizeof(float), 0, result_cols * sizeof(float), result_rows, stream));
	{
		dim3 grid, block;

		block.x = BLOCK_X;
		block.y = BLOCK_Y;
		block.z = 1;

		if (expected_cols >= 4)
		{
			grid.x = (expected_cols / 4 + block.x - 1) / block.x;
			grid.y = (expected_rows + block.y - 1) / block.y;
			grid.z = (batch_size + block.z - 1) / block.z;

			mean_square_error_float4_kernel << < grid, block, 0, stream >> > (
				batch_size, expected_rows, expected_cols/4,scale,
				batched_predicted, batched_predicted_stride,
				expected, expected_stride,
				result, result_stride);
		}
		else
		{
			grid.x = (expected_cols + block.x - 1) / block.x;
			grid.y = (expected_rows + block.y - 1) / block.y;
			grid.z = (batch_size + block.z - 1) / block.z;

			mean_square_error_naive_kernel << < grid, block, 0, stream >> > (
				batch_size, expected_rows, expected_cols,scale,
				batched_predicted, batched_predicted_stride,
				expected, expected_stride,
				result, result_stride);
		}

	


		checkCudaErrors(cudaGetLastError());
	}
}




__global__
static void batched_sgemv_kernel
(
	const int batch_size,
	float ** __restrict__ w, const int w_rows, const int w_cols, const int w_stride,
	float ** __restrict__ x, const int x_rows, const int x_cols, const int x_stride,
	float ** __restrict__ y, const int y_rows, const int  y_cols, const int y_stride
)

{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *w_ = w[batch];
		float *x_ = x[batch];
		float *y_ = y[batch];

		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < w_rows; row += gridDim.y * blockDim.y)
		{
			float sum = 0.0f;
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < w_cols; col += gridDim.x * blockDim.x)
			{				
				sum += dot(reinterpret_cast<float4 *>(&w_[row * w_stride])[col], reinterpret_cast<float4 *>(x_)[col]);
			}

			sum = warpReduceSum(sum);
			if ((threadIdx.x & 31) == 0)
			{
				atomicAdd(&y_[row], sum);
			}
		}
	}
}

__host__ 
static inline void batched_sgemv(
	const cudaStream_t &stream,
	const std::size_t & batch_size,
	float ** w, const std::size_t &w_rows, const std::size_t &w_cols, const std::size_t &w_stride,
	float ** x, const std::size_t &x_rows, const std::size_t &x_cols, const std::size_t &x_stride,
	float ** y, const std::size_t &y_rows, const std::size_t &y_cols, const std::size_t &y_stride)
{
	assert(x_rows == 1);
	assert(y_rows == 1);
	assert(x_cols == w_cols);
	assert(y_cols == w_rows);
	dim3 grid, block;

	block.x = BLOCK_X;
	block.y = BLOCK_Y;
	block.z = 1;

	grid.x = (w_cols / 4 + block.x - 1) / block.x;
	grid.y = (w_rows + block.y - 1) / block.y;
	grid.z = (batch_size + block.z - 1) / block.z;


	batched_sgemv_kernel << <grid, block, 0, stream >> >
		(
			batch_size, w, w_rows, w_cols/4, w_stride,
			x, x_rows, x_cols/4, x_stride,
			y, y_rows, y_cols, y_stride
			);
	checkCudaErrors(cudaGetLastError());

}

__global__
static void batched_reset_kernel(
	const int batch_size, const int rows, const int cols,
	 float ** __restrict__ x, const int x_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *X = x[batch];
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rows; row += gridDim.y * blockDim.y)
		{
#pragma unroll 8
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x)
			{
				reinterpret_cast<float4 *>(&X[row * x_stride])[col] = make_float4(0.0f);
			}
		}
	}
}

static inline void batched_reset(const cudaStream_t &stream,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	float **x, const std::size_t &x_stride)
{
	dim3 grid, block;
	block.x = BLOCK_X;
	block.y = BLOCK_Y;
	block.z = 1;
	grid.x = (cols / 4 + block.x - 1) / block.x;
	grid.y = (rows + block.y - 1) / block.y;
	grid.z = (batch_size + block.z - 1) / block.z;

	batched_reset_kernel << < grid, block, 0, stream >> > (
		batch_size, rows, cols/4, x, x_stride);

	checkCudaErrors(cudaGetLastError());
}


__global__
static void batched_update_reservoir_kernel(
	const int batch_size,
	const int t, const float leak_rate,
	 float ** __restrict__ u_ffwd, const int u_ffwd_rows, const int u_ffwd_cols, const int u_ffwd_stride,
	 float ** __restrict__ u, const int u_rows, const int u_cols, const int u_stride,

	float ** __restrict__ p, const int p_rows, const int p_cols, const int p_stride,
	float ** __restrict__ x_res, const int x_res_rows, const int x_res_cols, const int x_res_stride
)
{
	for (int batch = blockIdx.y * blockDim.y + threadIdx.y; batch < batch_size; batch += gridDim.y * blockDim.y)
	{
		float *P = p[batch];
		float *U = u[batch];
		float *X = x_res[batch];
		float *U_ffwd = u_ffwd[batch];

		float *u_ffwd_t = &U_ffwd[t * u_ffwd_stride];
		assert(t < u_ffwd_rows);
//#pragma unroll 8
		for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < x_res_cols ; col += gridDim.x * blockDim.x)
		{
			float4 _p = reinterpret_cast<float4 *>(P)[col];
			float4 _u = reinterpret_cast<float4 *>(U)[col];
			float4 _u_ffwd = reinterpret_cast<float4 *>(u_ffwd_t)[col];

			_p += leak_rate * ( _u_ffwd + _u - _p);
			reinterpret_cast<float4 *>(X)[col] = tanhf(_p);
			reinterpret_cast<float4 *>(P)[col] = _p;
		}
	}
}
__global__
static void batched_update_reservoir_no_input_kernel(
	const int batch_size,
	const int t, const float leak_rate,
	float ** __restrict__ u, const int u_rows, const int u_cols, const int u_stride,

	float ** __restrict__ p, const int p_rows, const int p_cols, const int p_stride,
	float ** __restrict__ x_res, const int x_res_rows, const int x_res_cols, const int x_res_stride
)
{
	for (int batch = blockIdx.y * blockDim.y + threadIdx.y; batch < batch_size; batch += gridDim.y * blockDim.y)
	{
		float *P = p[batch];
		float *U = u[batch];
		float *X = x_res[batch];

		for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < x_res_cols; col += gridDim.x * blockDim.x)
		{
			float4 _p = reinterpret_cast<float4 *>(P)[col];
			float4 _u = reinterpret_cast<float4 *>(U)[col];

			_p += leak_rate * (_u - _p);
			reinterpret_cast<float4 *>(X)[col] = tanhf(_p);
			reinterpret_cast<float4 *>(P)[col] = _p;
		}
	}
}
__host__
static inline void batched_update_reservoir_no_input
(
	const cudaStream_t &stream,
	const std::size_t &batch_size, const std::size_t t, const float &leak_rate, 

	float **u, const std::size_t &u_rows, const std::size_t &u_cols, const std::size_t &u_stride,
	float **p, const std::size_t &p_rows, const std::size_t &p_cols, const std::size_t &p_stride,
	float **x_res, const std::size_t &x_res_rows, const std::size_t &x_res_cols, const std::size_t &x_res_stride
)
{
	dim3 grid, block;

	block.x = 128;
	block.y = 1;

	grid.x = (x_res_cols / 4 + block.x - 1) / block.x;
	grid.y = (batch_size + block.y - 1) / block.y;

	batched_update_reservoir_no_input_kernel << < grid, block, 0, stream >> > (
		batch_size, t, leak_rate,
		
		u, u_rows, u_cols / 4, u_stride,

		p, p_rows, p_cols / 4, p_stride,
		x_res, x_res_rows, x_res_cols / 4, x_res_stride);

	checkCudaErrors(cudaGetLastError());
}

__host__
static inline void batched_update_reservoir
(
	const cudaStream_t &stream,
	const std::size_t &batch_size, const std::size_t t, const float &leak_rate, 
	 float **u_ffwd, const std::size_t &u_ffwd_rows, const std::size_t &u_ffwd_cols, const std::size_t &u_ffwd_stride,
	 float **u, const std::size_t &u_rows, const std::size_t &u_cols, const std::size_t &u_stride,
	float **p, const std::size_t &p_rows, const std::size_t &p_cols, const std::size_t &p_stride,
	float **x_res, const std::size_t &x_res_rows, const std::size_t &x_res_cols, const std::size_t &x_res_stride
)
{
	dim3 grid, block;

	block.x = 128;
	block.y = 1;

	grid.x = (x_res_cols / 4 + block.x - 1) / block.x;
	grid.y = (batch_size + block.y - 1) / block.y;

	batched_update_reservoir_kernel  << < grid, block, 0, stream >> > (
		batch_size, t, leak_rate,
		u_ffwd, u_ffwd_rows, u_ffwd_cols, u_ffwd_stride,
		u, u_rows, u_cols/4, u_stride,

		p, p_rows, p_cols/4, p_stride,
		x_res, x_res_rows, x_res_cols/4, x_res_stride);

	checkCudaErrors(cudaGetLastError());
}
__device__
static const char *dev_cudaGetErrorEnum(cudaError_t error)
{
	switch (error)
	{
		case cudaSuccess:
			return "cudaSuccess";

		case cudaErrorMissingConfiguration:
			return "cudaErrorMissingConfiguration";

		case cudaErrorMemoryAllocation:
			return "cudaErrorMemoryAllocation";

		case cudaErrorInitializationError:
			return "cudaErrorInitializationError";

		case cudaErrorLaunchFailure:
			return "cudaErrorLaunchFailure";

		case cudaErrorPriorLaunchFailure:
			return "cudaErrorPriorLaunchFailure";

		case cudaErrorLaunchTimeout:
			return "cudaErrorLaunchTimeout";

		case cudaErrorLaunchOutOfResources:
			return "cudaErrorLaunchOutOfResources";

		case cudaErrorInvalidDeviceFunction:
			return "cudaErrorInvalidDeviceFunction";

		case cudaErrorInvalidConfiguration:
			return "cudaErrorInvalidConfiguration";

		case cudaErrorInvalidDevice:
			return "cudaErrorInvalidDevice";

		case cudaErrorInvalidValue:
			return "cudaErrorInvalidValue";

		case cudaErrorInvalidPitchValue:
			return "cudaErrorInvalidPitchValue";

		case cudaErrorInvalidSymbol:
			return "cudaErrorInvalidSymbol";

		case cudaErrorMapBufferObjectFailed:
			return "cudaErrorMapBufferObjectFailed";

		case cudaErrorUnmapBufferObjectFailed:
			return "cudaErrorUnmapBufferObjectFailed";

		case cudaErrorInvalidHostPointer:
			return "cudaErrorInvalidHostPointer";

		case cudaErrorInvalidDevicePointer:
			return "cudaErrorInvalidDevicePointer";

		case cudaErrorInvalidTexture:
			return "cudaErrorInvalidTexture";

		case cudaErrorInvalidTextureBinding:
			return "cudaErrorInvalidTextureBinding";

		case cudaErrorInvalidChannelDescriptor:
			return "cudaErrorInvalidChannelDescriptor";

		case cudaErrorInvalidMemcpyDirection:
			return "cudaErrorInvalidMemcpyDirection";

		case cudaErrorAddressOfConstant:
			return "cudaErrorAddressOfConstant";

		case cudaErrorTextureFetchFailed:
			return "cudaErrorTextureFetchFailed";

		case cudaErrorTextureNotBound:
			return "cudaErrorTextureNotBound";

		case cudaErrorSynchronizationError:
			return "cudaErrorSynchronizationError";

		case cudaErrorInvalidFilterSetting:
			return "cudaErrorInvalidFilterSetting";

		case cudaErrorInvalidNormSetting:
			return "cudaErrorInvalidNormSetting";

		case cudaErrorMixedDeviceExecution:
			return "cudaErrorMixedDeviceExecution";

		case cudaErrorCudartUnloading:
			return "cudaErrorCudartUnloading";

		case cudaErrorUnknown:
			return "cudaErrorUnknown";

		case cudaErrorNotYetImplemented:
			return "cudaErrorNotYetImplemented";

		case cudaErrorMemoryValueTooLarge:
			return "cudaErrorMemoryValueTooLarge";

		case cudaErrorInvalidResourceHandle:
			return "cudaErrorInvalidResourceHandle";

		case cudaErrorNotReady:
			return "cudaErrorNotReady";

		case cudaErrorInsufficientDriver:
			return "cudaErrorInsufficientDriver";

		case cudaErrorSetOnActiveProcess:
			return "cudaErrorSetOnActiveProcess";

		case cudaErrorInvalidSurface:
			return "cudaErrorInvalidSurface";

		case cudaErrorNoDevice:
			return "cudaErrorNoDevice";

		case cudaErrorECCUncorrectable:
			return "cudaErrorECCUncorrectable";

		case cudaErrorSharedObjectSymbolNotFound:
			return "cudaErrorSharedObjectSymbolNotFound";

		case cudaErrorSharedObjectInitFailed:
			return "cudaErrorSharedObjectInitFailed";

		case cudaErrorUnsupportedLimit:
			return "cudaErrorUnsupportedLimit";

		case cudaErrorDuplicateVariableName:
			return "cudaErrorDuplicateVariableName";

		case cudaErrorDuplicateTextureName:
			return "cudaErrorDuplicateTextureName";

		case cudaErrorDuplicateSurfaceName:
			return "cudaErrorDuplicateSurfaceName";

		case cudaErrorDevicesUnavailable:
			return "cudaErrorDevicesUnavailable";

		case cudaErrorInvalidKernelImage:
			return "cudaErrorInvalidKernelImage";

		case cudaErrorNoKernelImageForDevice:
			return "cudaErrorNoKernelImageForDevice";

		case cudaErrorIncompatibleDriverContext:
			return "cudaErrorIncompatibleDriverContext";

		case cudaErrorPeerAccessAlreadyEnabled:
			return "cudaErrorPeerAccessAlreadyEnabled";

		case cudaErrorPeerAccessNotEnabled:
			return "cudaErrorPeerAccessNotEnabled";

		case cudaErrorDeviceAlreadyInUse:
			return "cudaErrorDeviceAlreadyInUse";

		case cudaErrorProfilerDisabled:
			return "cudaErrorProfilerDisabled";

		case cudaErrorProfilerNotInitialized:
			return "cudaErrorProfilerNotInitialized";

		case cudaErrorProfilerAlreadyStarted:
			return "cudaErrorProfilerAlreadyStarted";

		case cudaErrorProfilerAlreadyStopped:
			return "cudaErrorProfilerAlreadyStopped";

			/* Since CUDA 4.0*/
		case cudaErrorAssert:
			return "cudaErrorAssert";

		case cudaErrorTooManyPeers:
			return "cudaErrorTooManyPeers";

		case cudaErrorHostMemoryAlreadyRegistered:
			return "cudaErrorHostMemoryAlreadyRegistered";

		case cudaErrorHostMemoryNotRegistered:
			return "cudaErrorHostMemoryNotRegistered";

			/* Since CUDA 5.0 */
		case cudaErrorOperatingSystem:
			return "cudaErrorOperatingSystem";

		case cudaErrorPeerAccessUnsupported:
			return "cudaErrorPeerAccessUnsupported";

		case cudaErrorLaunchMaxDepthExceeded:
			return "cudaErrorLaunchMaxDepthExceeded";

		case cudaErrorLaunchFileScopedTex:
			return "cudaErrorLaunchFileScopedTex";

		case cudaErrorLaunchFileScopedSurf:
			return "cudaErrorLaunchFileScopedSurf";

		case cudaErrorSyncDepthExceeded:
			return "cudaErrorSyncDepthExceeded";

		case cudaErrorLaunchPendingCountExceeded:
			return "cudaErrorLaunchPendingCountExceeded";

		case cudaErrorNotPermitted:
			return "cudaErrorNotPermitted";

		case cudaErrorNotSupported:
			return "cudaErrorNotSupported";

			/* Since CUDA 6.0 */
		case cudaErrorHardwareStackError:
			return "cudaErrorHardwareStackError";

		case cudaErrorIllegalInstruction:
			return "cudaErrorIllegalInstruction";

		case cudaErrorMisalignedAddress:
			return "cudaErrorMisalignedAddress";

		case cudaErrorInvalidAddressSpace:
			return "cudaErrorInvalidAddressSpace";

		case cudaErrorInvalidPc:
			return "cudaErrorInvalidPc";

		case cudaErrorIllegalAddress:
			return "cudaErrorIllegalAddress";

			/* Since CUDA 6.5*/
		case cudaErrorInvalidPtx:
			return "cudaErrorInvalidPtx";

		case cudaErrorInvalidGraphicsContext:
			return "cudaErrorInvalidGraphicsContext";

		case cudaErrorStartupFailure:
			return "cudaErrorStartupFailure";

		case cudaErrorApiFailureBase:
			return "cudaErrorApiFailureBase";

			/* Since CUDA 8.0*/
		case cudaErrorNvlinkUncorrectable:
			return "cudaErrorNvlinkUncorrectable";
	}

	return "<unknown>";
}

template< typename T >
__device__
void dev_check(T result, char const *const func, const char *const file, int const line)
{
	if (result)
	{
		printf("CUDA error at %s:%d code=%d\"%s\" \n",
			file, line, static_cast<unsigned int>(result),  func);
			// Make sure we call CUDA Device Reset before exiting
	}
}
#define dev_checkCudaErrors(val)           dev_check ( (val), #val, __FILE__, __LINE__ )
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>
#include <sstream>




template <bool gather_states>
static void copy_states(
	const cudaStream_t &stream, const std::size_t &batch_size, const std::size_t &t, const std::size_t &ts,
	const std::size_t &stimulus_size,
	const std::size_t &reservoir_size,
	const std::size_t &prediction_size,
	const std::size_t &stimulus_stride,
	const std::size_t &reservoir_stride,
	const std::size_t &prediction_stride,
	const float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	const float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	const float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	const float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float *states, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{}

template <>
static void copy_states<true>(const cudaStream_t &stream, const std::size_t &batch_size, const std::size_t &t, const std::size_t &ts,
	const std::size_t &stimulus_size,
	const std::size_t &reservoir_size,
	const std::size_t &prediction_size,
	const std::size_t &stimulus_stride,
	const std::size_t &reservoir_stride,
	const std::size_t &prediction_stride,
	const float **batched_incoming, const std::size_t &batched_incoming_rows, const std::size_t &batched_incoming_cols, const std::size_t &batched_incoming_strides,
	const float **batched_expected, const std::size_t &batched_expected_rows, const std::size_t &batched_expected_cols, const std::size_t &batched_expected_strides,
	const float **batched_x_ro, const std::size_t &batched_x_ro_rows, const std::size_t &batched_x_ro_cols, const std::size_t &batched_x_ro_strides,
	const float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t &batched_x_res_strides,
	float *states, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride)
{	
	std::vector<float *> incoming_ptr(batch_size);
	std::vector<float *> expected_ptr(batch_size);
	std::vector<float *> x_ro_ptr(batch_size);
	std::vector<float *> x_res_ptr(batch_size);
	/*cudaStream_t incoming, expected, x_ro, x_res;*/
	/*cudaEvent_t incoming_terminated, expected_terminated, x_ro_terminated, x_res_terminated;
	checkCudaErrors(cudaStreamCreateWithFlags(&incoming, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&expected, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&x_ro, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&x_res, cudaStreamNonBlocking));*/
	checkCudaErrors(cudaMemcpyAsync(incoming_ptr.data(), batched_incoming, batch_size * sizeof(float *), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaMemcpyAsync(expected_ptr.data(), batched_expected, batch_size * sizeof(float *), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaMemcpyAsync(x_ro_ptr.data(), batched_x_ro, batch_size * sizeof(float *), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaMemcpyAsync(x_res_ptr.data(), batched_x_res, batch_size * sizeof(float *), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));

	/*checkCudaErrors(cudaEventCreate(&incoming_terminated));
	checkCudaErrors(cudaEventCreate(&expected_terminated));
	checkCudaErrors(cudaEventCreate(&x_ro_terminated));
	checkCudaErrors(cudaEventCreate(&x_res_terminated));

	checkCudaErrors(cudaStreamSynchronize(incoming));
	checkCudaErrors(cudaStreamSynchronize(expected));
	checkCudaErrors(cudaStreamSynchronize(x_ro));*/
	checkCudaErrors(cudaStreamSynchronize(stream));

	std::size_t offset = 0;
	float *states_ts = &states[ts * states_stride];

	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		std::size_t offset = 0;
		std::size_t  stimulus_col = batch * stimulus_stride + batch_size * offset;
		checkCudaErrors(cudaMemcpyAsync(
			states_ts + stimulus_col,
			&incoming_ptr[batch][t * batched_incoming_strides],
			sizeof(float) * stimulus_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
		offset += stimulus_stride;

		std::size_t  desired_col = batch * prediction_stride + batch_size * offset;
		checkCudaErrors(cudaMemcpyAsync(
			states_ts + desired_col,
			&expected_ptr[batch][t * batched_expected_strides],
			sizeof(float) * prediction_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
		offset += prediction_stride;

		std::size_t  reservoir_col = batch * reservoir_stride + batch_size * offset;
		checkCudaErrors(cudaMemcpyAsync(
			states_ts + reservoir_col,
			x_res_ptr[batch],
			sizeof(float) * reservoir_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
		offset += reservoir_stride;

		std::size_t  predicted_col = batch * prediction_stride + batch_size * offset;
		checkCudaErrors(cudaMemcpyAsync(
			states_ts + predicted_col,
			x_ro_ptr[batch],
			sizeof(float) * prediction_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
		offset += prediction_stride;
	}
	
}


template <bool overwrite_states>
static inline void initialize_states(const cudaStream_t &stream,  unsigned long &seed,
	const std::size_t &batch_size,
	float **batched_ptr, const std::size_t &batched_ptr_rows, const std::size_t &batched_ptr_cols, const std::size_t &batched_ptr_stride,
	const float &initial_state_scale)
{
}
template <>
static inline void initialize_states<true>(const cudaStream_t &stream, unsigned long &seed,
	const std::size_t &batch_size,
	float **batched_ptr, const std::size_t &batched_ptr_rows, const std::size_t &batched_ptr_cols, const std::size_t &batched_ptr_stride,
	const float &initial_state_scale)
{
	random_uniform(stream, seed, -initial_state_scale, initial_state_scale, 0.0f, batch_size, batched_ptr_rows, batched_ptr_cols, batched_ptr, batched_ptr_stride);
	seed += batch_size * batched_ptr_rows * batched_ptr_cols;
}

static inline void sgemm_nt(
	const cublasHandle_t handle, const int batch_size,
	const float alpha, const float beta,
	const float **a, const int a_rows, const int a_cols, const int a_stride,
	const float **b, const int b_rows, const int b_cols, const int b_stride,
	float **c, const int c_rows, const int c_cols, const int c_stride
	)
{
	auto op_a_rows = a_rows;
	auto op_a_cols = a_cols;
	auto op_b_rows = b_cols;
	auto op_b_cols = b_rows;

	assert(op_a_rows == c_rows);
	auto m = op_a_rows;
	assert(op_b_cols == c_cols);
	auto n = op_b_cols;
	assert(op_a_cols == op_b_rows);
	auto k = op_a_cols;

	checkCudaErrors(cublasSgemmBatched(handle,
		cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T,
		m,n,k,
		&alpha,
		a, a_stride,
		b, b_stride,
		&beta,
		c, c_stride,
		batch_size
	));
}

static inline void sgemm_tn(
	const cublasHandle_t handle, const int batch_size,
	const float alpha, const float beta,
	const float **a, const int a_rows, const int a_cols, const int a_stride,
	const float **b, const int b_rows, const int b_cols, const int b_stride,
	float **c, const int c_rows, const int c_cols, const int c_stride
)
{
	auto op_a_rows = a_cols;
	auto op_a_cols = a_rows;
	auto op_b_rows = b_rows;
	auto op_b_cols = b_cols;

	assert(op_a_rows == c_rows);
	auto m = op_a_rows;
	assert(op_b_cols == c_cols);
	auto n = op_b_cols;
	assert(op_a_cols == op_b_rows);
	auto k = op_a_cols;

	checkCudaErrors(cublasSgemmBatched(handle,
		cublasOperation_t::CUBLAS_OP_T, cublasOperation_t::CUBLAS_OP_N,
		m, n, k,
		&alpha,
		a, a_stride,
		b, b_stride,
		&beta,
		c, c_stride,
		batch_size
	));
}
static inline void sgemm_nn(
	const cublasHandle_t handle, const int batch_size,
	const float alpha, const float beta,
	const float **a, const int a_rows, const int a_cols, const int a_stride,
	const float **b, const int b_rows, const int b_cols, const int b_stride,
	float **c, const int c_rows, const int c_cols, const int c_stride
)
{
	auto op_a_rows = a_rows;
	auto op_a_cols = a_cols;
	auto op_b_rows = b_rows;
	auto op_b_cols = b_cols;

	assert(op_a_rows == c_rows);
	auto m = op_a_rows;
	assert(op_b_cols == c_cols);
	auto n = op_b_cols;
	assert(op_a_cols == op_b_rows);
	auto k = op_a_cols;

	checkCudaErrors(cublasSgemmBatched(handle,
		cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N,
		m, n, k,
		&alpha,
		a, a_stride,
		b, b_stride,
		&beta,
		c, c_stride,
		batch_size
	));
}

__global__
static void update_readout_error_kernel(
	const int batch_size,
	const int t, 
	const float learning_rate,
	 float ** __restrict__ batched_x_ro, const int batched_x_ro_rows, const int batched_x_ro_cols, const int batched_x_ro_stride,
	 float ** __restrict__ batched_expected, const int batched_expected_rows, const int batched_expected_cols, const int batched_expected_stride,
	float ** __restrict__ batched_error, const int batched_error_rows, const int batched_error_cols, const int batched_error_stride
)
{
	for (int batch = blockIdx.y * blockDim.y + threadIdx.y; batch < batch_size; batch += gridDim.y * blockDim.y)
	{
		float *E = batched_error[batch];
		 float *D = batched_expected[batch];
		 float *X = batched_x_ro[batch];

		for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < batched_error_cols >> 2; col += gridDim.x * blockDim.x)
		{
			float4 d = reinterpret_cast<float4 *>(&D[t * batched_expected_cols])[col];
			float4 x = reinterpret_cast<float4 *>(X)[col];
			float4 e = learning_rate * (d - x);
			/*assert(!isnan(d));
			assert(!isnan(x));
			assert(!isnan(e));*/
			reinterpret_cast<float4 *>(E)[col] = e;
		}
	}
}
#define BLOCK_X 32
#define BLOCK_Y 32


__global__
static void widrow_hoff_kernel(
	const int batch_size, 
	float **__restrict__  batched_w_ro, const int batched_w_ro_rows, const int batched_w_ro_cols, const int batched_w_ro_stride,
	float **__restrict__  batched_x_res, const int batched_x_res_rows, const int batched_x_res_cols, const int batched_x_res_stride,
	float **__restrict__  batched_error, const int batched_error_rows, const int batched_error_cols, const int batched_error_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *W = batched_w_ro[batch];
		float *E = batched_error[batch];
		float *X = batched_x_res[batch];

		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < batched_w_ro_rows; row += gridDim.y * blockDim.y)
		{
			const float e = E[row];
			//assert(!isnan(e));
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < batched_w_ro_cols >> 2; col += gridDim.x * blockDim.x)
			{
			
				//assert(!isnan(x));
				reinterpret_cast<float4 *>(&W[row * batched_w_ro_stride])[col] += reinterpret_cast<float4 *>(X)[col] * e;
			}
		}
	}

}
template <typename Parameter>
static inline void update_readout(
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const std::size_t &batch_size, const std::size_t & t, const Parameter &parameter,
	 float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t & batched_x_res_stride,
	 float **batched_x_ro, const std::size_t & batched_x_ro_rows, const std::size_t & batched_x_ro_cols, const std::size_t & batched_x_ro_stride,
	 float **batched_expected, const std::size_t & batched_expected_rows, const std::size_t & batched_expected_cols, const std::size_t &batched_expected_stride,
	float **batched_error, const std::size_t &batched_error_rows, const std::size_t &batched_error_cols, const std::size_t & batched_error_stride,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t & batched_w_ro_stride)
{

}



template <>
static inline void update_readout(
	const cudaStream_t &stream, 
	const cublasHandle_t &handle, 
	const std::size_t &batch_size, const std::size_t & t, const Widrow_Hoff &parameter,
	 float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t & batched_x_res_stride,
	 float **batched_x_ro, const std::size_t & batched_x_ro_rows, const std::size_t & batched_x_ro_cols, const std::size_t & batched_x_ro_stride,
	 float **batched_expected, const std::size_t & batched_expected_rows, const std::size_t & batched_expected_cols, const std::size_t &batched_expected_stride,
	float **batched_error, const std::size_t &batched_error_rows, const std::size_t &batched_error_cols, const std::size_t & batched_error_stride,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t & batched_w_ro_stride)
{
	assert(batched_x_res_rows == 1);
	assert(batched_x_ro_rows == 1);
	assert(t < batched_expected_rows);
	assert(batched_w_ro_cols == batched_x_res_cols);
	assert(batched_w_ro_rows == batched_x_ro_cols);

	{
		dim3 block;
		dim3 grid;

		block.x = warpSize * 4;
		grid.x = (batched_error_cols / 4+ block.x - 1) / block.x;

		block.y = 1;
		grid.y = (batch_size + block.y - 1) / block.y;

		update_readout_error_kernel << <grid, block, 0, stream >> >
			(
				batch_size, t, parameter.get_learning_rate(),
				batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_stride,
				batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_stride,
				batched_error, batched_error_rows, batched_error_cols, batched_error_stride
				);

		checkCudaErrors(cudaGetLastError());
	}
	

	{
		dim3 block;
		dim3 grid;

		block.x = BLOCK_X;
		block.y = BLOCK_Y;
		block.z = 1;
		grid.x = (batched_w_ro_cols / 4 + block.x - 1) / block.x;
		grid.y = (batched_w_ro_rows + block.y - 1) / block.y;
		grid.z = (batch_size + block.z - 1) / block.z;
		widrow_hoff_kernel << <grid, block, 0, stream >> >
			(
				batch_size,
				batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_stride,
				batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_stride,
				batched_error, batched_error_rows, batched_error_cols, batched_error_stride
				);

		checkCudaErrors(cudaGetLastError());
	}
}
static const float one = 1.0f;
static const float zero = 0.0f;

struct isnan_test {
	__host__ __device__ bool operator()(const float a) const {
		return isnan(a);
	}
};


__global__
static void isnan_kernel(
	const int batch_size,
	float **__restrict__  batched_x, const int batched_x_rows, const int batched_x_cols, const int batched_x_stride,
	bool *result
)
{
	*result = false;
	for (int batch = 0; batch < batch_size; batch++)
	{
		float *x = batched_x[batch];
		auto r = thrust::transform_reduce(thrust::seq, x, x + batched_x_rows * batched_x_stride, isnan_test(), false, thrust::plus<bool>());
		*result |= r;

	}
}

void isnan(
	const cudaStream_t &stream, 
	const int batch_size,
	float **__restrict__  batched_x, const int batched_x_rows, const int batched_x_cols, const int batched_x_stride)
{
	bool *dev_result = NULL;
	bool result = false;
	checkCudaErrors(cudaMalloc((void **)&dev_result, sizeof(bool)));
	isnan_kernel << <1, 1, 0, stream >> > (batch_size, batched_x, batched_x_rows, batched_x_cols, batched_x_stride, dev_result);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaMemcpyAsync(&result, dev_result, sizeof(bool), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaStreamSynchronize(stream));
	if (result)
		throw std::runtime_error("NAN");
}

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
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	)
{
	/*isnan(stream, batch_size, batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides);
	isnan(stream, batch_size, batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides);*/
	sgemm_tn(
		handle, batch_size,
		one, zero,
		(const float **)batched_w_ffwd, batched_w_ffwd_cols, batched_w_ffwd_rows, batched_w_ffwd_strides,
		(const float **)batched_incoming, batched_incoming_cols, batched_incoming_rows, batched_incoming_strides,
	

		batched_u_ffwd, batched_u_ffwd_cols, batched_u_ffwd_rows, batched_u_ffwd_strides
	);




	std::size_t ts = 0;
	for (std::size_t repetition = 0; repetition < repetitions; repetition++)
	{
		initialize_states<overwrite_states>(stream,  seed, batch_size,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides, initial_state_scale);
		//isnan(stream, batch_size, batched_p, batched_p_rows, batched_p_cols, batched_p_strides);
		initialize_states<overwrite_states>(stream, seed, batch_size,
			(float **)batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides, initial_state_scale);
		//isnan(stream, batch_size, batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides);
		for (std::size_t k = 0; k < durations[repetition]; k++, ts++)
		{
			int t = offsets[ts];
	
			//std::cout << "t = " << t << std::endl;
			batched_reset(stream, batch_size, batched_u_rows, batched_u_cols, batched_u, batched_u_strides);
			//isnan(stream, batch_size, batched_u, batched_u_rows, batched_u_cols, batched_u_strides);
			batched_sgemv(stream, batch_size,
				batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
				batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
				batched_u, batched_u_rows, batched_u_cols, batched_u_strides
			);
			//isnan(stream, batch_size, batched_u, batched_u_rows, batched_u_cols, batched_u_strides);

			if (t < 0)
			{
				t = -t;
				batched_update_reservoir_no_input(
					stream,
					batch_size, t, leak_rate,
					batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
					batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
					batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides);
			}
			else
			{
				batched_update_reservoir(
					stream,
					batch_size, t, leak_rate,
					batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
					batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
					batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
					batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides);
			}


			//isnan(stream, batch_size, batched_p, batched_p_rows, batched_p_cols, batched_w_ffwd_strides);
			//isnan(stream, batch_size, batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides);

			batched_reset(stream, batch_size, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro, batched_x_ro_strides);
			//isnan(stream, batch_size, batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides);
			batched_sgemv(stream, batch_size,
				batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
				batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
				batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides
			);
			//isnan(stream, batch_size, batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides);
			update_readout<Parameter>(
				stream, handle, batch_size, t, parameter,
				batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
				batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
				batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
				batched_error, batched_error_rows, batched_error_cols, batched_error_strides,
				batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides);
			//isnan(stream, batch_size, batched_error, batched_error_rows, batched_error_cols, batched_error_strides);
			//isnan(stream, batch_size, batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides);
			copy_states<gather_states>(stream, batch_size, t, ts,
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
__global__
static void sum_inplace_kernel(
	const int batch_size, const int place_cells_number, const int size,
	float   *** __restrict__ batched_hypothesis)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cells_number; place_cell += gridDim.y * blockDim.y)
		{
			const int place_cell_a = place_cell;
			const int place_cell_b = place_cell + place_cells_number;
			float *hypothesis_a = batched_hypothesis[batch][place_cell_a];
			float *hypothesis_b = batched_hypothesis[batch][place_cell_b];
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x)
			{
				reinterpret_cast<float4 *>(hypothesis_a)[idx] += reinterpret_cast<float4 *>(hypothesis_b)[idx];
			}
		}
	}
}
__global__
static void sum_kernel(
	const int batch_size, const int place_cells_number, const int size,
	float   *** __restrict__ batched_hypothesis,
	float   ** __restrict__ batched_location)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *location = batched_location[batch];
		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cells_number; place_cell += gridDim.y * blockDim.y)
		{
			const int place_cell_a = place_cell;
			const int place_cell_b = place_cell + place_cells_number;
			float *hypothesis_a = batched_hypothesis[batch][place_cell_a];
			float *hypothesis_b = batched_hypothesis[batch][place_cell_b];
		
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x)
			{
				reinterpret_cast<float4 *>(location)[idx] = reinterpret_cast<float4 *>(hypothesis_a)[idx] + reinterpret_cast<float4 *>(hypothesis_b)[idx];
			}
		}
	}
}
__global__
static void weighted_acc_inplace_kernel(
	const int batch_size, const int place_cells_number, const int size,
	float *** __restrict__ batched_hypothesis,
	float ** __restrict__ batched_scale,
	float ** __restrict__ batched_location_probability)
{
	assert(place_cells_number == 1);
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *location_probability = batched_location_probability[batch];
		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cells_number; place_cell += gridDim.y * blockDim.y)
		{
			const float scale = batched_scale[batch][place_cell];
			float *hypothesis = batched_hypothesis[batch][place_cell];
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x)
			{
			
				reinterpret_cast<float4 *>(location_probability)[idx] += reinterpret_cast<float4 *>(hypothesis)[idx] / scale;
					
			}
		}
	}
}
__global__
static void weighted_sum_inplace_kernel( 
	const int batch_size,
	const int place_cells_number, const int size,
	float   ** __restrict__ scale, const int scale_stride,
	float   ***__restrict__ batched_hypothesis, const int hypothesis_stride
	)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cells_number; place_cell += gridDim.y * blockDim.y)
		{
			const int place_cell_a = place_cell;
			const int place_cell_b = place_cell + place_cells_number;
			float *hypothesis_a = batched_hypothesis[batch][place_cell_a];
			float *hypothesis_b = batched_hypothesis[batch][place_cell_b];
			const float scale_a = scale[batch][place_cell_a];
			const float scale_b = scale[batch][place_cell_b];
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x)
			{
				reinterpret_cast<float4 *>(hypothesis_a)[idx] =
					reinterpret_cast<float4 *>(hypothesis_a)[idx] / scale_a +
					reinterpret_cast<float4 *>(hypothesis_b)[idx] / scale_b;
			}
		}
	}
}


__global__
static void location_hypothesis_kernel(
	const int batch_size,
	const int place_cells_number, const int size, const float minus_inv_sigma2,
	 const float   ** __restrict__ batched_firing_rate_map, const int firing_rate_stride,
	 const float  ** __restrict__ batched_prediction, const int prediction_stride,
	float  *** __restrict__ batched_hypothesis_map, const int hypothesis_stride,
	float  ** __restrict__ batched_scale, const int scale_stride)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *scale = batched_scale[batch];
		const float *prediction = batched_prediction[batch];
	

		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cells_number; place_cell += gridDim.y * blockDim.y)
		{
			const float *firing_rate_map = batched_firing_rate_map[place_cell];
			float *hypothesis_map = batched_hypothesis_map[batch][place_cell];
			const float p = prediction[place_cell];

			float4 sum4 = make_float4(0.0f);
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < size; idx += gridDim.x * blockDim.x)
			{
				const float4 value = reinterpret_cast<float4 *>(const_cast<float *>(firing_rate_map))[idx] - p;
				const float4 response = expf(value * value * minus_inv_sigma2);
				sum4 += response;
				reinterpret_cast<float4 *>(hypothesis_map)[idx] = response;
			}

			float sum = sum4.x + sum4.y + sum4.z + sum4.w;
			sum = warpReduceSum(sum);
			if ((threadIdx.x & 31) == 0)
			{
				atomicAdd(&scale[place_cell], sum * place_cells_number);
			}
		}
	
	}
}






void compute_place_cell_location_probability(
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float &sigma,
	const float ** firing_rate_map, const std::size_t &firing_rate_map_rows, const std::size_t &firing_rate_map_cols, const std::size_t &firing_rate_map_stride,
	float **scale, const std::size_t &scale_rows, const std::size_t &scale_cols, const std::size_t &scale_stride,
	const float **prediction, const std::size_t &prediction_rows, const std::size_t &prediction_cols, const std::size_t &prediction_stride,
	float *** hypothesis_map, const std::size_t &hypothesis_map_rows, const std::size_t &hypothesis_map_cols, const std::size_t &hypothesis_map_stride,
	float ** location_probability, const std::size_t &location_probability_rows, const std::size_t &location_probability_cols, const std::size_t &location_probability_stride) // batch_size * 2

{
	
	const auto minus_sigma2_inv = -1.0/(sigma * sigma);
	const std::size_t size = rows * firing_rate_map_stride;
	dim3 block, grid;
	block.x = 32;
	block.y = 32;
	block.z = 1;
	batched_reset(stream, batch_size, scale_rows, scale_cols, scale, scale_stride);
	grid.x = (size / 4 + block.x - 1) / block.x;
	grid.z = (batch_size + block.z - 1) / block.z;
	{
		grid.y = (place_cells_number + block.y - 1) / block.y;
		// reset location hyppothesis
		location_hypothesis_kernel << <grid, block, 0, stream >> > (
				batch_size, place_cells_number, size / 4, minus_sigma2_inv,
				firing_rate_map, firing_rate_map_stride,
				prediction, prediction_stride,
				hypothesis_map, hypothesis_map_stride,
				scale, scale_stride);
		checkCudaErrors(cudaGetLastError());
	}

	std::size_t place_cells_number_range = place_cells_number / 2;
	const std::size_t place_cells_number_remaining = place_cells_number - place_cells_number_range * 2;

	{
		grid.y = (place_cells_number_range + block.y - 1) / block.y;

		weighted_sum_inplace_kernel << <grid, block, 0, stream >> > (
			batch_size, place_cells_number_range, size/4,
			scale, scale_stride,
			hypothesis_map, hypothesis_map_stride);
		checkCudaErrors(cudaGetLastError());
	}

	while (place_cells_number_range >= 2)
	{
		place_cells_number_range /= 2;

		grid.y = (place_cells_number_range + block.y - 1) / block.y;

		sum_inplace_kernel << <grid, block, 0, stream >> > (
			batch_size, place_cells_number_range, size/4,
			hypothesis_map);
		checkCudaErrors(cudaGetLastError());
	}

	assert(place_cells_number_range == 1);
	{
		grid.y = (place_cells_number_range + block.y - 1) / block.y;

		sum_kernel << <grid, block, 0, stream >> > (
			batch_size, place_cells_number_range, size/4,
			hypothesis_map, 
			location_probability);
		checkCudaErrors(cudaGetLastError());
	}


	{
		if (place_cells_number_remaining > 0)
		{
			grid.y = (place_cells_number_range + block.y - 1) / block.y;

			weighted_acc_inplace_kernel << <grid, block, 0, stream >> > (
				batch_size, place_cells_number_range, size/4,
				hypothesis_map,
				scale,
				location_probability);
			checkCudaErrors(cudaGetLastError());
		}
	}
}

__global__
static void inside_circle_kernel(
	const int batch_size, const int rows, const int cols, const int location_probability_stride,
	const float radius2,
	const float scale,
	const float   * __restrict__ x_grid,
	const float   * __restrict__ y_grid,
	const float ** __restrict__ batched_current_location,
	float   ** __restrict__ batched_location_probability)

{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *location_probability = batched_location_probability[batch];
		const float *current_location = batched_current_location[batch];
		const float x = current_location[0];
		const float y = current_location[1];
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rows; row += gridDim.y * blockDim.y)
		{
			const float dy = y_grid[row] - y;
			const float dy2 = dy * dy;
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x)
			{
				const float4 dx = reinterpret_cast<float4 *>(const_cast<float *>(x_grid))[col] - x;
				const float4 dx2 = dx * dx;
				const float4 lhs = dx2 + dy2;
				float4 p = reinterpret_cast<float4 *>(&location_probability[row * location_probability_stride])[col];

				curandStatePhilox4_32_10_t state;
		
				// seed a random number generator
				curand_init(col * rows + row + batch * rows * cols, 0, 0, &state);

				p += curand_uniform4(&state) * scale;
				float4 s;
				s.x = lhs.x > radius2 ? 0.0f : p.x;
				s.y = lhs.y > radius2 ? 0.0f : p.y;
				s.z = lhs.z > radius2 ? 0.0f : p.z;
				s.w = lhs.w > radius2 ? 0.0f : p.w;

				reinterpret_cast<float4 *>(&location_probability[row * location_probability_stride])[col] = s;
			}
		}
	
	}
}



void compute_reachable_locations(
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float &radius, const float &scale,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float **batched_current_location, const std::size_t &batched_current_location_rows, const std::size_t &batched_current_location_cols, const std::size_t &batched_current_location_stride,
	float **batched_x_grid_centered2, const std::size_t &batched_x_grid_centered2_rows, const std::size_t &batched_x_grid_centered2_cols, const std::size_t &batched_x_grid_centered2_stride,
	float **batched_y_grid_centered2, const std::size_t &batched_y_grid_centered2_rows, const std::size_t &batched_y_grid_centered2_cols, const std::size_t &batched_y_grid_centered2_stride,
	float  **batched_location_probability, const std::size_t &batched_location_probability_rows, const std::size_t &batched_location_probability_cols, const std::size_t &batched_location_probability_strides)
{

	dim3 grid, block;
	block.x = BLOCK_X;
	block.y = BLOCK_Y;
	block.z = 1;

	grid.x = (batched_location_probability_cols / 4 + block.x - 1) / block.x;
	grid.y = (batched_location_probability_rows  + block.y - 1) / block.y;
	grid.z = (batch_size + block.z - 1) / block.z;

		inside_circle_kernel << <grid, block, 0, stream >> > (batch_size, rows, cols / 4, batched_location_probability_strides, radius * radius, scale,
			x_grid, y_grid, batched_current_location, batched_location_probability);
		checkCudaErrors(cudaGetLastError());


}




struct normalize_functor
{
	const float scale;
	const float offset;
	__device__
	normalize_functor(float a, float b) : scale(1.0f/ (b - a)) , offset(-a)
	{}
	__device__
	float operator()(const float& x) const
	{ 
		return (x + offset ) * scale; 
	} 
};





__global__
void select_location_kernel(const int batch_size, 
	const float * __restrict__ x_grid, const int x_grid_rows, const int x_grid_cols, const int x_grid_stride,
	const float *__restrict__ y_grid, const int y_grid_rows, const int y_grid_cols, const int y_grid_stride,
	float **__restrict__ batched_location_probability, const int batched_location_probability_rows, const int batched_location_probability_cols, const int batched_location_probability_stride,
	float ** __restrict__ batched_predicted_location, const int batched_predicted_location_rows, const int batched_predicted_location_cols, const int batched_predicted_location_stride

)
{
#ifndef __DEBUG
	for (int batch = blockIdx.x * blockDim.x + threadIdx.x; batch < batch_size; batch += gridDim.x * blockDim.x)
	{


		float *predicted_position = batched_predicted_location[batch];
		float *location_probability = batched_location_probability[batch];
		int idx = -1;
		{
			cublasHandle_t handle;
			dev_checkCudaErrors(cublasCreate(&handle));
			dev_checkCudaErrors(cublasIsamax(handle, batched_location_probability_stride * batched_location_probability_rows, location_probability, 1, &idx));
			dev_checkCudaErrors(cublasDestroy(handle));
		}

		int row = idx / batched_location_probability_stride;
		int col = idx % batched_location_probability_stride;
		predicted_position[0] = x_grid[col];
		predicted_position[1] = y_grid[row];
	}
#endif
}



void compute_select_most_probable_location(const cudaStream_t &stream,  const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t &batched_location_probability_rows, const std::size_t &batched_location_probability_cols, const std::size_t &batched_location_probability_strides,
	float **batched_predicted_location, const std::size_t &batched_predicted_location_rows, const std::size_t &batched_predicted_location_cols, const std::size_t &batched_predicted_location_strides
)
{
	std::vector<float *> batched_location_probability_ptr(batch_size);
	std::vector<float *> batched_predicted_location_ptr(batch_size);
	std::vector<cub::KeyValuePair <int, float> *> argmax_ptr(batch_size);
	std::vector<cub::KeyValuePair <int, float>> argmax_host(batch_size);
	std::vector<void *> temp_storage_ptr(batch_size);
	std::vector<std::size_t> temp_storage_bytes(batch_size);
	std::vector<cudaStream_t> child_streams(batch_size);



	checkCudaErrors(cudaMemcpyAsync(batched_location_probability_ptr.data(), batched_location_probability, batch_size * sizeof(float *), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));
	checkCudaErrors(cudaMemcpyAsync(batched_predicted_location_ptr.data(), batched_predicted_location, batch_size * sizeof(float *), cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));



	for (int batch = 0; batch < batch_size; batch++)
	{
		float *d_in = batched_location_probability_ptr[batch];
		checkCudaErrors(cudaStreamCreateWithFlags(&child_streams[batch], cudaStreamNonBlocking));

		checkCudaErrors(cudaMalloc(&argmax_ptr[batch], sizeof(cub::KeyValuePair <int, float>)));
		cub::DeviceReduce::ArgMax(temp_storage_ptr[batch], temp_storage_bytes[batch], d_in, argmax_ptr[batch], batched_location_probability_strides * batched_location_probability_rows, child_streams[batch]);
		// Allocate temporary storage
		checkCudaErrors(cudaMalloc(&temp_storage_ptr[batch], temp_storage_bytes[batch]));
		cub::DeviceReduce::ArgMax(temp_storage_ptr[batch], temp_storage_bytes[batch], d_in, argmax_ptr[batch], batched_location_probability_strides * batched_location_probability_rows, child_streams[batch]);
		checkCudaErrors(cudaMemcpyAsync(&argmax_host[batch], argmax_ptr[batch], sizeof(cub::KeyValuePair <int, float>), cudaMemcpyKind::cudaMemcpyDeviceToHost, child_streams[batch]));
	}


	for (int batch = 0; batch < batch_size; batch++)
	{
		
		checkCudaErrors(cudaStreamSynchronize(child_streams[batch]));
		checkCudaErrors(cudaStreamDestroy(child_streams[batch]));
		checkCudaErrors(cudaFree(temp_storage_ptr[batch]));
		checkCudaErrors(cudaFree(argmax_ptr[batch]));

		auto idx = argmax_host[batch].key;
		auto row = idx / batched_location_probability_strides;
		auto col = idx % batched_location_probability_strides;
		checkCudaErrors(cudaMemcpyAsync(batched_predicted_location_ptr[batch] + 0, &x_grid[col], sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
		checkCudaErrors(cudaMemcpyAsync(batched_predicted_location_ptr[batch] + 1, &y_grid[row], sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
		
	}
}


__global__
void reduce_cols_kernel(const int batch_size,
	float **__restrict__ batched_location_probability, const int batched_location_probability_rows, const int batched_location_probability_cols, const int batched_location_probability_stride,
	float **__restrict__ batched_reduced_location_probability, const int batched_reduced_location_probability_rows, const int batched_reduced_location_probability_cols, const int batched_reduced_location_probability_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *location_proabability = batched_location_probability[batch];
		float *reduced_location_probability = batched_reduced_location_probability[batch];

		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < batched_location_probability_rows; row += gridDim.y * blockDim.y)
		{
			float sum = 0.0f;
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < batched_location_probability_cols >> 2; col += gridDim.x * blockDim.x)
			{
				float4 x = reinterpret_cast<float4 *>(&location_proabability[row * batched_location_probability_stride])[col];
				sum += x.x + x.y + x.z + x.w;
			}
			sum = warpReduceSum(sum);
			if ((threadIdx.x & 31) == 0)
			{
				atomicAdd(&reduced_location_probability[row], sum);
			}
		}
	}
}

static inline void reduce_cols(const cudaStream_t &stream, const std::size_t &batch_size,
	const float  **batched_location_probability, const std::size_t &batched_location_probability_rows, const std::size_t &batched_location_probability_cols, const std::size_t &batched_location_probability_strides,
	float **batched_reduced_location_probability, const std::size_t &batched_reduced_location_probability_rows, const std::size_t &batched_reduced_location_probability_cols, const std::size_t &batched_reduced_location_probability_strides
)
{
	dim3 grid, block;
	block.x = BLOCK_X;
	block.y = BLOCK_Y;
	block.z = 1;

	grid.x = (batched_location_probability_cols / 4 + block.x - 1) / block.x;
	grid.y = (batched_location_probability_rows + block.y - 1) / block.y;
	grid.z = (batch_size + block.z - 1) / block.z;
	reduce_cols_kernel << <grid, block, 0, stream >> > (batch_size,
		(float  **)batched_location_probability, batched_location_probability_rows, batched_location_probability_cols, batched_location_probability_strides,
		batched_reduced_location_probability, batched_reduced_location_probability_rows, batched_reduced_location_probability_cols, batched_reduced_location_probability_strides);
	checkCudaErrors(cudaGetLastError());
}

__global__
void draw_location_kernel(const int batch_size, const unsigned long seed,
	const float * __restrict__ x_grid, const int x_grid_rows, const int x_grid_cols, const int x_grid_stride,
	const float *__restrict__ y_grid, const int y_grid_rows, const int y_grid_cols, const int y_grid_stride,
	float **__restrict__ batched_location_probability, const int batched_location_probability_rows, const int batched_location_probability_cols, const int batched_location_probability_stride,
	float **__restrict__ batched_reduced_location_probability, const int batched_reduced_location_probability_rows, const int batched_reduced_location_probability_cols, const int batched_reduced_location_probability_stride,
	float **__restrict__ batched_row_cumsum, const int batched_row_cumsum_rows, const int batched_row_cumsum_cols, const int batched_row_cumsum_stride,
	float **__restrict__ batched_col_cumsum, const int batched_col_cumsum_rows, const int batched_col_cumsum_cols, const int batched_col_cumsum_stride,
	float ** __restrict__ batched_predicted_location, const int batched_predicted_location_rows, const int batched_predicted_location_cols, const int batched_predicted_location_stride

)
{
#ifndef __DEBUG
	curandState localState;
	curand_init(seed, 0, 0, &localState);

	for (int batch = 0; batch < batch_size; batch++)
	{
		float *location_probability = batched_location_probability[batch];
		float *reduced_location_probability = batched_reduced_location_probability[batch];
		float *row_cumsum = batched_row_cumsum[batch];
		float *col_cumsum = batched_col_cumsum[batch];
		float *predicted_position = batched_predicted_location[batch];

		thrust::inclusive_scan(thrust::seq, reduced_location_probability, reduced_location_probability + batched_location_probability_rows, row_cumsum);

		const float row_a = row_cumsum[0];
		const float row_b = row_cumsum[batched_location_probability_rows - 1];
		//printf("before %f %f\n", row_cumsum[0], row_cumsum[batched_location_probability_rows - 1]);
		thrust::transform(thrust::seq, row_cumsum, row_cumsum + batched_location_probability_rows, row_cumsum, normalize_functor(row_a, row_b));
		//printf("after %f %f\n", row_cumsum[0], row_cumsum[batched_location_probability_rows - 1]);
		float ry = curand_uniform(&localState);

		int row = thrust::distance(row_cumsum, thrust::lower_bound(thrust::seq, row_cumsum, row_cumsum + batched_location_probability_rows, ry));
		//printf("ry = %f -> row =  %d\n", ry, row);
		assert(0 <= row && row < batched_location_probability_rows);

		float *location_probability_row = &location_probability[row * batched_location_probability_stride];
		thrust::inclusive_scan(thrust::seq, location_probability_row, location_probability_row + batched_location_probability_cols, col_cumsum);
		const float col_a = col_cumsum[0];
		const float col_b = col_cumsum[batched_location_probability_cols - 1];
		//printf("before %f %f\n", col_cumsum[0], col_cumsum[batched_location_probability_cols- 1]);
		thrust::transform(thrust::seq, col_cumsum, col_cumsum + batched_location_probability_cols, col_cumsum, normalize_functor(col_a, col_b));
		//printf("after %f %f\n", col_cumsum[0], col_cumsum[batched_location_probability_cols - 1]);
		float rx = curand_uniform(&localState);

		int col = thrust::distance(col_cumsum, thrust::lower_bound(thrust::seq, col_cumsum, col_cumsum + batched_location_probability_cols, rx));
		// printf("rx = %f -> col = %d\n", rx, col);
		assert(0 <= col && col < batched_location_probability_cols);

		predicted_position[0] = x_grid[col];
		predicted_position[1] = y_grid[row];
	}
#endif
}

void compute_draw_probable_location(
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	const float  **batched_location_probability, const std::size_t &batched_location_probability_rows, const std::size_t &batched_location_probability_cols, const std::size_t &batched_location_probability_strides,
	float **batched_reduced_location_probability, const std::size_t &batched_reduced_location_probability_rows, const std::size_t &batched_reduced_location_probability_cols, const std::size_t &batched_reduced_location_probability_strides,
	float **batched_row_cumsum, const std::size_t &batched_row_cumsum_rows, const std::size_t &batched_row_cumsum_cols, const std::size_t &batched_row_cumsum_strides,
	float **batched_col_cumsum, const std::size_t &batched_col_cumsum_rows, const std::size_t &batched_col_cumsum_cols, const std::size_t &batched_col_cumsum_strides,
	float **batched_predicted_location, const std::size_t &batched_predicted_location_rows, const std::size_t &batched_predicted_location_cols, const std::size_t &batched_predicted_location_stride
)
{
	batched_reset(stream, batch_size, batched_reduced_location_probability_rows, batched_reduced_location_probability_cols,
		batched_reduced_location_probability, batched_reduced_location_probability_strides);

	reduce_cols(stream, batch_size,
		batched_location_probability, batched_location_probability_rows, batched_location_probability_cols, batched_location_probability_strides,
		batched_reduced_location_probability, batched_reduced_location_probability_rows, batched_reduced_location_probability_cols, batched_reduced_location_probability_strides);

	static unsigned long seed = 0;
	draw_location_kernel << <1, 1, 0, stream >> > (batch_size, seed,
		x_grid, x_grid_rows, x_grid_cols, x_grid_stride,
		y_grid, y_grid_rows, y_grid_cols, y_grid_stride,
		(float  **)batched_location_probability, batched_location_probability_rows, batched_location_probability_cols, batched_location_probability_strides,
		batched_reduced_location_probability, batched_reduced_location_probability_rows, batched_reduced_location_probability_cols, batched_reduced_location_probability_strides,
		batched_row_cumsum, batched_row_cumsum_rows, batched_row_cumsum_cols, batched_row_cumsum_strides,
		batched_col_cumsum, batched_col_cumsum_rows, batched_col_cumsum_cols, batched_col_cumsum_strides,
		batched_predicted_location, batched_predicted_location_rows, batched_predicted_location_cols, batched_predicted_location_stride
		);
	checkCudaErrors(cudaGetLastError());
	seed += batch_size * 2;
}



template  void update_model<true, true, Widrow_Hoff>(
	const std::size_t &batch_size,
	unsigned long &seed,
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const Widrow_Hoff &parameter,
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
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	);
template void update_model< true, false, Widrow_Hoff >(
	const std::size_t &batch_size,
	unsigned long &seed,
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const Widrow_Hoff &parameter,
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
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	);
template void update_model< false, true, Widrow_Hoff>(
	const std::size_t &batch_size,
	unsigned long &seed,
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const Widrow_Hoff &parameter,
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
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	);
template void update_model< false, false, Widrow_Hoff>(
	const std::size_t &batch_size,
	unsigned long &seed,
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const Widrow_Hoff &parameter,
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
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	);
template void update_model<true, true, Nothing>(
	const std::size_t &batch_size,
	unsigned long &seed,
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const Nothing &parameter,
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
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	);
template void update_model< true, false, Nothing >(
	const std::size_t &batch_size,
	unsigned long &seed,
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const Nothing &parameter,
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
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	);
template void update_model< false, true, Nothing>(
	const std::size_t &batch_size,
	unsigned long &seed,
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const Nothing &parameter,
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
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	);
template void update_model< false, false, Nothing>(
	const std::size_t &batch_size,
	unsigned long &seed,
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const Nothing &parameter,
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
	const int *offsets, const int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	);

