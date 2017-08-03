#include "stdafx.h"

#include <cuda.h>

#include <cuda_runtime.h>
#include "Algorithm.cuh"
#include "Random.cuh"

#include <helper_cuda.h>
#include <helper_math.h>

#define warpSize 32

__device__
static inline void atomicAdd(float4* a, float4 b)
{
	float * addr = &a->x;

	atomicAdd(addr + 0, b.x);
	atomicAdd(addr + 4, b.y);
	atomicAdd(addr + 8, b.z);
	atomicAdd(addr + 12, b.w);
}

__device__
static inline float4 __shfl_down(float4 a, int offset)
{
	float4 b;

	b.x = __shfl_down(a.x, offset);
	b.y = __shfl_down(a.y, offset);
	b.z = __shfl_down(a.z, offset);
	b.w = __shfl_down(a.w, offset);

	return b;
}

template <typename Type>
__device__ __inline__
static Type warpReduceSum(Type val) {
	for (int offset = warpSize /2 ; offset > 0; offset /= 2)
		val += __shfl_down(val, offset);
	return val;
}
template <typename Type>
__inline__ __device__
static Type blockReduceSum(Type val) 
{
	static __shared__ Type shared[warpSize]; // Shared mem for 32 partial sums
	int lane = threadIdx.x & (warpSize-1);
	int wid = threadIdx.x >> 5;

	val = warpReduceSum(val);     // Each warp performs partial reduction

	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

								  //read from shared memory only if that warp existed
	val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

	if (wid == 0) 
		val = warpReduceSum(val); //Final reduce within first warp

	return val;
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
static void mean_square_error_kernel(
	const int batch_size, const int rows, const int cols,
	const float ** __restrict__ predicted, const int predicted_stride, 
	const float * __restrict__  expected,const int expected_stride,
	float *__restrict__  result, const int result_stride

)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rows; row += gridDim.y * blockDim.y)
		{
			float sum = 0.0f;
			for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x)
			{
				const float d = predicted[batch][col * predicted_stride + row] - expected[col * expected_stride + row];

				sum += (d * d) / cols;
			}

			sum = warpReduceSum(sum);
			if (threadIdx.x % warpSize == 0)
				atomicAdd(&result[batch * result_stride + row], sum);
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
	checkCudaErrors(cudaMemset2DAsync(result, result_stride * sizeof(float), 0, result_rows * sizeof(float), result_cols, stream));
	{
		dim3 grid, block;
		TRN::GPU::Context::get_block_dims(mean_square_error_kernel, { expected_cols, expected_rows, batch_size}, grid, block);

		mean_square_error_kernel << < grid, block, 0, stream >> > (
			batch_size, expected_rows, expected_cols,
			batched_predicted, batched_predicted_stride,
			 expected, expected_stride,
			result, result_stride);
		checkCudaErrors(cudaGetLastError());
	}
}

__device__
static inline float dp(float4 a, float4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

template<const int row_factor = 1, const int col_factor = 1>
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
		float *X = x[batch];
		float *Y = y[batch];
		float *W = w[batch];



		for (int col = col_factor * blockIdx.y * blockDim.y + threadIdx.y; col < w_cols; col += col_factor * gridDim.y * blockDim.y)
		{
			float sum[col_factor] = { 0.0f };

#pragma unroll col_factor
			for (int k = 0; k < col_factor; k++)
			{
				
				/*float sum1 = 0.0f;
				float sum2 = 0.0f;
				float sum3 = 0.0f;*/

					for (int row = row_factor*blockIdx.x * blockDim.x + threadIdx.x; row < w_rows / (4 * row_factor); row += row_factor*gridDim.x * blockDim.x)
					{
#pragma unroll row_factor
						for (int l = 0; l < row_factor; l++)
						{
							sum[k] += dot(reinterpret_cast<float4 *>(&W[(col + blockDim.y * k) * w_stride])[row + blockDim.x * l], reinterpret_cast<float4 *>(X)[row + blockDim.x * l]);
						}
						sum[k] = warpReduceSum(sum[k]);
						/*sum1 = warpReduceSum(sum1);
						sum2 = warpReduceSum(sum2);
						sum3 = warpReduceSum(sum3);*/
				
					}
			
			
			}
			if ((threadIdx.x & (warpSize - 1)) == 0)
			{
#pragma unroll col_factor
				for (int k = 0; k < col_factor; k++)
				{
					atomicAdd(&Y[col + blockDim.y * k], sum[k]);
				}
				/*atomicAdd(&Y[col + blockDim.y * 1], sum1);
				atomicAdd(&Y[col + blockDim.y * 2], sum2);
				atomicAdd(&Y[col + blockDim.y * 3], sum3);*/
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
	assert(x_cols == 1);
	assert(w_rows == x_rows);
	assert(w_cols == y_rows);
	assert(y_cols == 1);
	dim3 grid, block;



	block.x = 32;
	block.y = 32;
	block.z =1;

	//auto shared_size = block.x * block.y * sizeof(float);
	auto W_cols = w_cols;
	 /*if (W_cols % 8 == 0)
	{
		grid.x = (w_rows / (4*8) + block.x - 1) / block.x;
		grid.y = (w_cols / 8 + block.y - 1) / block.y;
		grid.z = (batch_size + block.z - 1) / block.z;
		batched_sgemv_kernel<8> << < grid, block, 0, stream >> > (
			batch_size,
			w, w_rows, w_cols, w_stride,
			x, x_rows, x_cols, x_stride,
			y, y_rows, y_cols, y_stride);
	}
	*/

	const int rf = 4;
	const int cf = 4;
	grid.x = (w_rows / (4*rf) + block.x - 1) / block.x;
	grid.y = (w_cols / cf + block.y - 1) / block.y;
	grid.z = (batch_size + block.z - 1) / block.z;
	batched_sgemv_kernel<rf, cf> << < grid, block, 0, stream >> > (
			batch_size,
			w, w_rows, w_cols, w_stride,
			x, x_rows, x_cols, x_stride,
			y, y_rows, y_cols, y_stride);
	
	
	checkCudaErrors(cudaGetLastError());
}

template<const int factor = 1>
__global__
static void batched_reset_kernel(
	const int batch_size, const int rows, const int cols,
	 float ** __restrict__ x, const int x_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *X = x[batch];
		for (int col = blockIdx.y * blockDim.y + threadIdx.y; col < cols; col += gridDim.y * blockDim.y)
		{
#pragma unroll factor
			for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < rows >> 2; row += gridDim.x * blockDim.x)
			{
				reinterpret_cast<float4 *>(&X[col * x_stride])[row] = make_float4(0.0f);
			}
		}
	}
}

static inline void batched_reset(const cudaStream_t &stream,
	const std::size_t &batch_size, const std::size_t &rows, const std::size_t &cols,
	float **x, const std::size_t &x_stride)
{
	dim3 grid, block;
	auto Rows = rows / 4;
	block.x = 32;
	grid.x = (Rows + block.x - 1) / block.x;

	block.y = 32;
	grid.y = (cols + block.y - 1) / block.y;

	block.z = 1;
	grid.z = (batch_size + block.z - 1) / block.z;
	if (Rows % 8 == 0)
	{
		batched_reset_kernel <8> << < grid, block, 0, stream >> > (
			 batch_size, rows, cols, x, x_stride);
	}
	else if (Rows % 4 == 0)
	{
		batched_reset_kernel <4> << < grid, block, 0, stream >> > (
			 batch_size, rows, cols, x, x_stride);
	}
	else if (Rows % 2 == 0)
	{
		batched_reset_kernel <2> << < grid, block, 0, stream >> > (
			 batch_size, rows, cols, x, x_stride);
	}
	else
	{
		batched_reset_kernel <1> << < grid, block, 0, stream >> > (
			 batch_size, rows, cols, x, x_stride);
	}

	checkCudaErrors(cudaGetLastError());
}

static inline __device__
float4 tanhf(const float4 &a)
{
	float4 b;

	b.x = tanh(a.x);
	b.y = tanh(a.y);
	b.z = tanh(a.z);
	b.w = tanh(a.w);
}

template<const int factor = 1>
__global__
static void batched_update_reservoir_kernel(
	const int batch_size,
	const float leak_rate, const int t,
	const float ** __restrict__ u_ffwd, const int u_ffwd_rows, const int u_ffwd_cols, const int u_ffwd_stride,
	const float ** __restrict__ u, const int u_rows, const int u_cols, const int u_stride,

	float ** __restrict__ p, const int p_rows, const int p_cols, const int p_stride,
	float ** __restrict__ x_res, const int x_res_rows, const int x_res_cols, const int x_res_stride
)
{
	for (int batch = blockIdx.y * blockDim.y + threadIdx.y; batch < batch_size; batch += gridDim.y * blockDim.y)
	{
		float *P = p[batch];
		const float *U = u[batch];
		float *X = x_res[batch];
		const float *U_ffwd = u_ffwd[batch];
#pragma unroll factor
		for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < x_res_rows; row += gridDim.x * blockDim.x)
		{
			reinterpret_cast<float4 *>(P)[row] += leak_rate * (reinterpret_cast<const float4 *>(U)[row] + reinterpret_cast<const float4 *>(&U[t * u_ffwd_stride])[row] - reinterpret_cast<float4 *>(P)[row]);
			reinterpret_cast<float4 *>(X)[row] = tanhf(reinterpret_cast<float4 *>(P)[row]);
		}
	}
}

__host__
static inline void batched_update_reservoir
(
	const cudaStream_t &stream,
	const std::size_t &batch_size, const std::size_t t, const float &leak_rate,
	const float **u_ffwd, const std::size_t &u_ffwd_rows, const std::size_t &u_ffwd_cols, const std::size_t &u_ffwd_stride,
	const float **u, const std::size_t &u_rows, const std::size_t &u_cols, const std::size_t &u_stride,
	float **p, const std::size_t &p_rows, const std::size_t &p_cols, const std::size_t &p_stride,
	float **x_res, const std::size_t &x_res_rows, const std::size_t &x_res_cols, const std::size_t &x_res_stride
)
{
	dim3 grid, block;
	const std::size_t XResRows = x_res_rows / 4;
	block.x = 32;
	grid.x = (XResRows + block.x - 1) / block.x;

	block.y = 32;
	grid.y = (batch_size + block.y - 1) / block.y;


	if (XResRows % 8 == 0)
	{
		batched_update_reservoir_kernel <8> << < grid, block, 0, stream >> > (
			batch_size, t, leak_rate,
			u_ffwd, u_ffwd_rows / 4, u_ffwd_cols, u_ffwd_stride,
			u, u_rows / 4, u_cols, u_stride,

			p, p_rows / 4, p_cols, p_stride,
			x_res, x_res_rows / 4, x_res_cols, x_res_stride);
	}
	else if (XResRows % 4 == 0)
	{
		batched_update_reservoir_kernel <4> << < grid, block, 0, stream >> > (
			batch_size, t, leak_rate,
			u_ffwd, u_ffwd_rows / 4, u_ffwd_cols, u_ffwd_stride,
			u, u_rows / 4, u_cols, u_stride,

			p, p_rows / 4, p_cols, p_stride,
			x_res, x_res_rows / 4, x_res_cols, x_res_stride);
	}
	else if (XResRows % 2 == 0)
	{
		batched_update_reservoir_kernel <2> << < grid, block, 0, stream >> > (
			batch_size, t, leak_rate,
			u_ffwd, u_ffwd_rows / 4, u_ffwd_cols, u_ffwd_stride,
			u, u_rows / 4, u_cols, u_stride,

			p, p_rows / 4, p_cols, p_stride,
			x_res, x_res_rows / 4, x_res_cols, x_res_stride);
	}
	else
	{
		batched_update_reservoir_kernel <1> << < grid, block, 0, stream >> > (
			batch_size, t, leak_rate,
			u_ffwd, u_ffwd_rows / 4, u_ffwd_cols, u_ffwd_stride,
			u, u_rows / 4, u_cols, u_stride,

			p, p_rows / 4, p_cols, p_stride,
			x_res, x_res_rows / 4, x_res_cols, x_res_stride);
	}
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
		printf("CUDA error at %s:%d code=%d(%s) \"%s\" \n",
			file, line, static_cast<unsigned int>(result), dev_cudaGetErrorEnum(result), func);
			// Make sure we call CUDA Device Reset before exiting
	}
}
#define dev_checkCudaErrors(val)           dev_check ( (val), #val, __FILE__, __LINE__ )


__global__
void copy_transpose(
	const int col, const int row,
	const float *__restrict__ x, const int size,
	float *__restrict__ states, const int states_stride
)
{
	for (int k = blockIdx.x * blockDim.x + threadIdx.x; k < size; k += gridDim.x * blockDim.x)
		states[(col + k) * states_stride + row] = x[k];
}
__global__
void copy_states_kernel(const int batch_size, const int t, const int ts,
	const int stimulus_size,
	const int reservoir_size,
	const int prediction_size,
	const int stimulus_stride,
	const int reservoir_stride,
	const int prediction_stride,
	const float **__restrict__ batched_incoming, const int batched_incoming_rows, const int batched_incoming_cols, const int batched_incoming_strides,
	const float **__restrict__ batched_expected, const int batched_expected_rows, const int batched_expected_cols, const int batched_expected_strides,
	const float **__restrict__ batched_x_ro, const int batched_x_ro_rows, const int batched_x_ro_cols, const int batched_x_ro_strides,
	const float **__restrict__ batched_x_res, const int batched_x_res_rows, const int batched_x_res_cols, const int batched_x_res_strides,
	float *__restrict__ states, const int states_rows, const int states_cols, const int states_stride
)
{
	/*cudaStream_t *streams;

	dev_checkCudaErrors(cudaMalloc(&streams, batch_size * sizeof(cudaStream_t)));
	for (int batch = 0; batch < batch_size; batch++)
	{
		dev_checkCudaErrors(cudaStreamCreateWithFlags(&streams[batch], cudaStreamNonBlocking));
	}*/
	/*cudaStream_t stimulus, desired, predicted, reservoir;
	dev_checkCudaErrors(cudaStreamCreateWithFlags(&stimulus, cudaStreamNonBlocking));
	dev_checkCudaErrors(cudaStreamCreateWithFlags(&desired, cudaStreamNonBlocking));
	dev_checkCudaErrors(cudaStreamCreateWithFlags(&predicted, cudaStreamNonBlocking));
	dev_checkCudaErrors(cudaStreamCreateWithFlags(&reservoir, cudaStreamNonBlocking));
	for (int batch = 0; batch < batch_size; batch++)

	{
		int offset = 0;
		int stimulus_col = batch * stimulus_stride + batch_size * offset;
		dev_checkCudaErrors(cudaMemcpy2DAsync(
			&states[stimulus_col * states_stride + ts], states_stride * sizeof(float),
			&batched_incoming[batch][t], batched_incoming_strides * sizeof(float), 
			sizeof(float), stimulus_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice), stimulus);
		offset += stimulus_stride;
		int desired_col = batch * prediction_stride + batch_size * offset;
		dev_checkCudaErrors(cudaMemcpy2DAsync(
			&states[desired_col * states_stride + ts], states_stride * sizeof(float),
			&batched_expected[batch][t], batched_expected_strides * sizeof(float),
			sizeof(float), prediction_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice), desired);

		offset += prediction_stride;
		int reservoir_col = batch * reservoir_stride + batch_size * offset;
		*/
		/*{
			dim3 grid, block;

			block.x = 128;
			grid.x = (reservoir_size + block.x - 1) / block.x;
			copy_transpose << <grid, block, 0, reservoir >> > (
				reservoir_col, ts,
				batched_x_res[batch], reservoir_size,
				states, states_stride);
		}*/


		/*offset += prediction_stride;
		int predicted_col = batch * prediction_stride + batch_size * offset;
		*/
		/*{
			dim3 grid, block;

			block.x = 128;
			grid.x = (prediction_size + block.x - 1) / block.x;
			copy_transpose << <grid, block, 0, predicted >> > (predicted_col, ts, batched_x_ro[batch], prediction_size, states, states_stride);
		}*/

	/*dev_checkCudaErrors(cudaMemcpy2DAsync(
			&states[predicted_col * states_stride + ts], states_stride * sizeof(float),
			&batched_x_ro[batch][0],  sizeof(float) * prediction_size,
			sizeof(float) , prediction_size, cudaMemcpyKind::cudaMemcpyDeviceToDevice), desired);*/

		/*dev_checkCudaErrors(cudaMemcpy2DAsync(
			&states[reservoir_col * states_stride + ts], states_stride * sizeof(float),
			&batched_x_res[batch][0], batched_x_res_strides * sizeof(float),
			sizeof(float) , batched_x_res_rows, cudaMemcpyKind::cudaMemcpyDeviceToDevice), reservoir);*/

	/*}
	dev_checkCudaErrors(cudaStreamDestroy(stimulus));
	dev_checkCudaErrors(cudaStreamDestroy(desired));
	dev_checkCudaErrors(cudaStreamDestroy(predicted));
	dev_checkCudaErrors(cudaStreamDestroy(reservoir));*/
	/*for (int batch = 0; batch < batch_size; batch++)
	{
	
	}
	dev_checkCudaErrors(cudaFree(streams));*/
}
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
	copy_states_kernel << < 1, 1, 0, stream >> > (batch_size, t, ts,
		stimulus_size, reservoir_size, prediction_size,
		stimulus_stride, reservoir_stride, prediction_stride,
		batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
		batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
		batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
		batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
		states, states_rows, states_cols, states_stride);
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



template <const int factor = 1>
__global__
static void update_readout_error_kernel(
	const int batch_size,
	const int t, 
	const float learning_rate,
	const float ** __restrict__ batched_x_ro, const int batched_x_ro_rows, const int batched_x_ro_cols, const int batched_x_ro_stride,
	const float ** __restrict__ batched_expected, const int batched_expected_rows, const int batched_expected_cols, const int batched_expected_stride,
	float ** __restrict__ batched_error, const int batched_error_rows, const int batched_error_cols, const int batched_error_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *E = batched_error[batch];
		const float *D = batched_expected[batch];
		const float *X = batched_x_ro[batch];
#pragma unroll factor
		for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < batched_error_rows; row += gridDim.x * blockDim.x)
		{
			reinterpret_cast<float4 *>(E)[row] = learning_rate * (reinterpret_cast<const float4 *>(&D[row * batched_error_stride])[t] - reinterpret_cast<const float4 *>(X)[row]);
		}
	}
}
#define BLOCK_X 32
#define BLOCK_Y 32

template <const int factor = 1>
__global__
static void widrow_hoff_kernel(
	const int batch_size, 
	float **__restrict__  batched_w_ro, const int batched_w_ro_rows, const int batched_w_ro_cols, const int batched_w_ro_stride,
	const float **__restrict__  batched_x_res, const int batched_x_res_rows, const int batched_x_res_cols, const int batched_x_res_stride,
	const float **__restrict__  batched_error, const int batched_error_rows, const int batched_error_cols, const int batched_error_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		float *W = batched_w_ro[batch];
		const float *E = batched_error[batch];
		const float *X = batched_x_res[batch];

		for (int col = blockIdx.y * blockDim.y + threadIdx.y; col < batched_w_ro_cols >> 2; col += gridDim.y * blockDim.y)
		{
			const float4 &e = reinterpret_cast<const float4 *>(E)[col];
#pragma unroll factor
			for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < batched_w_ro_rows >> 4; row += gridDim.x * blockDim.x)
			{
				const float4 &x = reinterpret_cast<const float4 *>(X)[row];

				float4 exx = reinterpret_cast<float4 *>(&W[(col + 0) * batched_w_ro_stride])[row];
				float4 exy = reinterpret_cast<float4 *>(&W[(col + 1) * batched_w_ro_stride])[row];
				float4 exz = reinterpret_cast<float4 *>(&W[(col + 2) * batched_w_ro_stride])[row];
				float4 exw = reinterpret_cast<float4 *>(&W[(col + 3) * batched_w_ro_stride])[row];

				exx.x += e.x * x.x; exx.y += e.y * x.x; exx.z += e.z * x.x; exx.w += e.w * x.x;
				exy.x += e.x * x.y; exy.x += e.y * x.y; exy.z += e.z * x.y; exy.w += e.w * x.y;
				exz.x += e.x * x.z; exz.y += e.y * x.z; exz.z += e.z * x.z; exz.w += e.w * x.z;
				exw.x += e.x * x.w; exw.y += e.y * x.w; exw.z += e.z * x.w; exw.w += e.w * x.w;
				 
				reinterpret_cast<float4 *>(&W[(col + 0) * batched_w_ro_stride])[row] = exx;
				reinterpret_cast<float4 *>(&W[(col + 1) * batched_w_ro_stride])[row] = exy;
				reinterpret_cast<float4 *>(&W[(col + 2) * batched_w_ro_stride])[row] = exz;
				reinterpret_cast<float4 *>(&W[(col + 3) * batched_w_ro_stride])[row] = exw;
			}
		}
	}

}
template <typename Parameter>
static inline void update_readout(
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const std::size_t &batch_size, const std::size_t & t, const Parameter &parameter,
	const float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t & batched_x_res_stride,
	const float **batched_x_ro, const std::size_t & batched_x_ro_rows, const std::size_t & batched_x_ro_cols, const std::size_t & batched_x_ro_stride,
	const float **batched_expected, const std::size_t & batched_expected_rows, const std::size_t & batched_expected_cols, const std::size_t &batched_expected_stride,
	float **batched_error, const std::size_t &batched_error_rows, const std::size_t &batched_error_cols, const std::size_t & batched_error_stride,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t & batched_w_ro_stride)
{

}



template <>
static inline void update_readout(
	const cudaStream_t &stream, 
	const cublasHandle_t &handle, 
	const std::size_t &batch_size, const std::size_t & t, const Widrow_Hoff &parameter,
	const float **batched_x_res, const std::size_t &batched_x_res_rows, const std::size_t &batched_x_res_cols, const std::size_t & batched_x_res_stride,
	const float **batched_x_ro, const std::size_t & batched_x_ro_rows, const std::size_t & batched_x_ro_cols, const std::size_t & batched_x_ro_stride,
	const float **batched_expected, const std::size_t & batched_expected_rows, const std::size_t & batched_expected_cols, const std::size_t &batched_expected_stride,
	float **batched_error, const std::size_t &batched_error_rows, const std::size_t &batched_error_cols, const std::size_t & batched_error_stride,
	float **batched_w_ro, const std::size_t &batched_w_ro_rows, const std::size_t &batched_w_ro_cols, const std::size_t & batched_w_ro_stride)
{
	assert(batched_x_res_cols == 1);
	assert(batched_x_ro_cols == 1);
	assert(t < batched_expected_rows);
	assert(batched_w_ro_rows == batched_x_res_rows);
	assert(batched_w_ro_cols == batched_x_ro_rows);

	auto error_rows = batched_error_rows / 4;
	{
		dim3 block;
		dim3 grid;

		block.x = 1024;
		grid.x = (batched_error_rows + block.x - 1) / block.x;

		block.y = 1;
		grid.y = (batched_error_cols + block.y - 1) / block.y;

		block.z = 1;
		grid.z = (batch_size + block.z - 1) / block.z;

		if (error_rows % 8 == 0)
		{
			update_readout_error_kernel <8> << <grid, block, 0, stream >> >
				(
					batch_size, t, parameter.get_learning_rate(),
					batched_x_ro, batched_x_ro_rows / 4, batched_x_ro_cols, batched_x_ro_stride,
					batched_expected, batched_expected_rows, batched_expected_cols/4, batched_expected_stride,
					batched_error, batched_error_rows / 4, batched_error_cols, batched_error_stride
					);
		}
		else if (error_rows % 4 == 0)
		{
			update_readout_error_kernel <4> << <grid, block, 0, stream >> >
				(
					batch_size, t, parameter.get_learning_rate(),
					batched_x_ro, batched_x_ro_rows / 4, batched_x_ro_cols, batched_x_ro_stride,
					batched_expected, batched_expected_rows, batched_expected_cols/4, batched_expected_stride,
					batched_error, batched_error_rows / 4, batched_error_cols, batched_error_stride
					);
		}
		else if (error_rows % 2 == 0)
		{
			update_readout_error_kernel <2> << <grid, block, 0, stream >> >
				(
					batch_size, t, parameter.get_learning_rate(),
					batched_x_ro, batched_x_ro_rows / 4, batched_x_ro_cols, batched_x_ro_stride,
					batched_expected, batched_expected_rows, batched_expected_cols/4, batched_expected_stride,
					batched_error, batched_error_rows / 4, batched_error_cols, batched_error_stride
				);
		}
		else
		{
			update_readout_error_kernel <1> << <grid, block, 0, stream >> >
				(
					batch_size, t, parameter.get_learning_rate(),
					batched_x_ro, batched_x_ro_rows / 4, batched_x_ro_cols, batched_x_ro_stride,
					batched_expected, batched_expected_rows, batched_expected_cols/4, batched_expected_stride,
					batched_error, batched_error_rows / 4, batched_error_cols, batched_error_stride
					);
		}
		checkCudaErrors(cudaGetLastError());
	}
	
	auto w_ro_rows = batched_w_ro_rows / 16;
	auto w_ro_cols = batched_w_ro_cols / 4;
	{
		dim3 block;
		dim3 grid;
		
		block.x = 32;
		grid.x = (w_ro_rows + block.x - 1) / block.x;

		block.y = 32;
		grid.y = (w_ro_cols + block.y - 1) / block.y;

		block.z = 1;
		grid.z = (batch_size + block.z - 1) / block.z;

		if (w_ro_rows % 8 == 0)
		{
			widrow_hoff_kernel <8> << <grid, block, 0, stream >> >
				(
					batch_size,
					batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_stride,
					batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_stride,
					(const float **)batched_error, batched_error_rows, batched_error_cols, batched_error_stride
					);
		}
		else if (w_ro_rows % 4 == 0)
		{
			widrow_hoff_kernel <4> << <grid, block, 0, stream >> >
				(
					batch_size,
					batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_stride,
					batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_stride,
					(const float **)batched_error, batched_error_rows, batched_error_cols, batched_error_stride
					);
		}
		else if (w_ro_rows % 2 == 0)
		{
			widrow_hoff_kernel <2> << <grid, block, 0, stream >> >
				(
					batch_size,
					batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_stride,
					batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_stride,
					(const float **)batched_error, batched_error_rows, batched_error_cols, batched_error_stride
					);
		}
		else
		{
			widrow_hoff_kernel <1> << <grid, block, 0, stream >> >
				(
					batch_size,
					batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_stride,
					batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_stride,
					(const float **)batched_error, batched_error_rows, batched_error_cols, batched_error_stride
					);
		}
		checkCudaErrors(cudaGetLastError());
	}
	
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
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	)
{
	static const float one = 1.0f;
	static const float zero = 0.0f;
	sgemm_nt(
		handle, batch_size,
		one, zero,
		(const float **)batched_w_ffwd, batched_w_ffwd_rows, batched_w_ffwd_cols, batched_w_ffwd_strides,
		(const float **)batched_incoming, batched_incoming_rows, batched_incoming_cols, batched_incoming_strides,
		batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols,  batched_u_ffwd_strides
	);

	std::size_t ts = 0;
	for (std::size_t repetition = 0; repetition < repetitions; repetition++)
	{
		const int t0 = offsets[repetition];
		const int tn = t0 + durations[repetition];

		initialize_states<overwrite_states>(stream,  seed, batch_size,
			batched_p, batched_p_rows, batched_p_cols, batched_p_strides, initial_state_scale);
		initialize_states<overwrite_states>(stream, seed, batch_size,
			(float **)batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides, initial_state_scale);
	
		for (int t = t0; t < tn; t++, ts++)
		{
		
			batched_reset(stream, batch_size, batched_u_rows, batched_u_cols, batched_u, batched_u_strides);
		
			batched_sgemv(stream, batch_size,
				batched_w_in, batched_w_in_rows, batched_w_in_cols, batched_w_in_strides,
				batched_x_in, batched_x_in_rows, batched_x_in_cols, batched_x_in_strides,
				batched_u, batched_u_rows, batched_u_cols, batched_u_strides
			);
			continue;
			batched_update_reservoir(
				stream,
				batch_size, t, leak_rate,
				(const float **)batched_u_ffwd, batched_u_ffwd_rows, batched_u_ffwd_cols, batched_u_ffwd_strides,
				(const float **)batched_u, batched_u_rows, batched_u_cols, batched_u_strides,
				batched_p, batched_p_rows, batched_p_cols, batched_p_strides,
				batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides);
	
			batched_reset(stream, batch_size, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro, batched_x_ro_strides);
			batched_sgemv(stream, batch_size,
				batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides,
				batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
				batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides
			);
			update_readout<Parameter>(
				stream, handle, batch_size, t, parameter,
				(const float **)batched_x_res, batched_x_res_rows, batched_x_res_cols, batched_x_res_strides,
				(const float **)batched_x_ro, batched_x_ro_rows, batched_x_ro_cols, batched_x_ro_strides,
				(const float **)batched_expected, batched_expected_rows, batched_expected_cols, batched_expected_strides,
				batched_error, batched_error_rows, batched_error_cols, batched_error_strides,
				batched_w_ro, batched_w_ro_rows, batched_w_ro_cols, batched_w_ro_strides);
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
	const int batch_size, const int place_cells_number, const int rows, const int cols,
	float   *** __restrict__ hypothesis_map, const int hypothesis_map_stride)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cells_number; place_cell += gridDim.y * blockDim.y)
		{
			const int place_cell_a = place_cell;
			const int place_cell_b = place_cell + place_cells_number;
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols; idx += gridDim.x * blockDim.x)
			{
				int row = idx % rows;
				int col = idx / rows;
				hypothesis_map[batch][place_cell_a][col * hypothesis_map_stride + row] +=
					hypothesis_map[batch][place_cell_b][col * hypothesis_map_stride + row];
			}
		}
	}
}
__global__
static void sum_kernel(
	const int batch_size, const int place_cells_number, const int rows, const int cols,
	const float   *** __restrict__ hypothesis_map, const int hypothesis_stride,
	float   ** location, const int location_stride)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cells_number; place_cell += gridDim.y * blockDim.y)
		{
			const int place_cell_a = place_cell;
			const int place_cell_b = place_cell + place_cells_number;
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols; idx += gridDim.x * blockDim.x)
			{
				int row = idx % rows;
				int col = idx / rows;
				location[batch][col * location_stride + row] =
					hypothesis_map[batch][place_cell_a][col * hypothesis_stride + row] +
					hypothesis_map[batch][place_cell_b][col * hypothesis_stride + row];
			}
		}
	}
}
__global__
static void weighted_acc_inplace_kernel(
	const int batch_size, const int place_cells_number, const int rows, const int  cols,
	float ***hypothesis_map, const int hypothesis_map_stride,
	float **scale, const int scale_stride,
	float **location_probability, const int location_probability_stride)
{
	assert(place_cells_number == 1);
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cells_number; place_cell += gridDim.y * blockDim.y)
		{
			
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols; idx += gridDim.x * blockDim.x)
			{
				int row = idx % rows;
				int col = idx / rows;

				location_probability[batch][col * location_probability_stride + row] += hypothesis_map[batch][place_cell][col * hypothesis_map_stride + row] /
					scale[batch][scale_stride * place_cell];
			}
		}
	}
}
__global__
static void weighted_sum_inplace_kernel( 
	const int batch_size,
	const int place_cells_number, const int rows, const int cols,
	const float   ** __restrict__ scale, const int scale_stride,
	float   ***__restrict__ hypothesis, const int hypothesis_stride
	)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cells_number; place_cell += gridDim.y * blockDim.y)
		{
			const int place_cell_a = place_cell;
			const int place_cell_b = place_cell + place_cells_number;
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols; idx += gridDim.x * blockDim.x)
			{
				int row = idx % rows;
				int col = idx / rows;

				hypothesis[batch][place_cell_a][col * hypothesis_stride + row] =
					hypothesis[batch][place_cell_a][col * hypothesis_stride + row] / scale[batch][place_cell_a * scale_stride] +
					hypothesis[batch][place_cell_b][col * hypothesis_stride + row] / scale[batch][place_cell_b * scale_stride];
			}
		}
	}
}


__global__
static void location_hypothesis_kernel(
	const int batch_size,
	const int place_cells_number, const int rows, const int cols, const float minus_inv_sigma2,
	const float   ** __restrict__ firing_rate_map, const int firing_rate_stride,
	const float  ** __restrict__ prediction, const int prediction_stride,
	float  *** __restrict__ hypothesis_map, const int hypothesis_stride, 
	float  ** __restrict__ scale, const int scale_stride)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cells_number; place_cell += gridDim.y * blockDim.y)
		{
			float sum = 0.0f;
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols; idx += gridDim.x * blockDim.x)
			{
				int row = idx % rows;
				int col = idx / rows;

				const float value = firing_rate_map[place_cell][row + firing_rate_stride * col] - prediction[batch][place_cell * prediction_stride];
				const float response = expf(value * value * minus_inv_sigma2);
				sum += response;
				hypothesis_map[batch][place_cell][row + hypothesis_stride * col] = response;
			}
			sum = warpReduceSum(sum);
			if (threadIdx.x & (warpSize -1) == 0)
			{
				atomicAdd(&scale[batch][place_cell * scale_stride], sum * place_cells_number);
			}
		}
	
	}
}



__global__
static void compute_grid_centered_kernel(const float   * __restrict__ grid, const int cols, const float current, float   * __restrict__ grid_centered2)
{
#pragma unroll
	for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x)
	{
		const float v = grid[col] - current;
		grid_centered2[col] = v * v;
	}
}



__global__
static void filter_move_kernel(float   * __restrict__ location_probability, const float   * __restrict__ x_grid_centered2, const float   * __restrict__ y_grid_centered2,
	const float *x_grid, const float *y_grid,
	const float radius2,
	const int rows, const int cols, const int location_probability_stride)
{
	for (int row = blockIdx.y * blockDim.y + threadIdx.y; row < rows; row += gridDim.y * blockDim.y)
	{
		for (int col = blockIdx.x * blockDim.x + threadIdx.x; col < cols; col += gridDim.x * blockDim.x)
		{
			if (y_grid_centered2[row] + x_grid_centered2[col] > radius2)
			{
				location_probability[row * location_probability_stride + col] = 0.0f;
			}
		}
	}
}


__global__
static void sum_location_hypothesis_kernel(
	const int batch_size, const int place_cell_number, const int rows, const int cols,
	float *** __restrict__ hypothesis_map, const int hypothesis_map_stride,
	const float ** __restrict__ scale, const int scale_stride,
	float   ** __restrict__ location_probability, const int location_probability_stride
)
{
	for (int batch = blockIdx.z * blockDim.z + threadIdx.z; batch < batch_size; batch += gridDim.z * blockDim.z)
	{
		for (int place_cell = blockIdx.y * blockDim.y + threadIdx.y; place_cell < place_cell_number; place_cell += gridDim.y * blockDim.y)
		{
			for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < rows * cols; idx += gridDim.x * blockDim.x)
			{
				int row = idx % rows;
				int col = idx / rows;
			
				location_probability[batch][col * location_probability_stride + row] +=
					hypothesis_map[batch][place_cell][col * hypothesis_map_stride + row] / scale[batch][place_cell * scale_stride];		
			}
		}
	}
}


void compute_place_cell_location_probability(
	const cudaStream_t &stream,
	const cublasHandle_t &handle,
	const std::size_t &batch_size, const std::size_t &place_cells_number, const std::size_t &rows, const std::size_t &cols,
	const float &sigma,
	const float &radius,
	const float ** firing_rate_map, const std::size_t &firing_rate_map_rows, const std::size_t &firing_rate_map_cols, const std::size_t &firing_rate_map_stride,
	const float *x_grid, const std::size_t &x_grid_rows, const std::size_t &x_grid_cols, const std::size_t &x_grid_stride,
	const float *y_grid, const std::size_t &y_grid_rows, const std::size_t &y_grid_cols, const std::size_t &y_grid_stride,
	float **x_grid_centered2, const std::size_t &x_grid_centered2_rows, const std::size_t &x_grid_centered2_cols, const std::size_t &x_grid_centered2_stride,
	float **y_grid_centered2, const std::size_t &y_grid_centered2_rows, const std::size_t &y_grid_centered2_cols, const std::size_t &y_grid_centered2_stride,
	float **scale, const std::size_t &scale_rows, const std::size_t &scale_cols, const std::size_t &scale_stride,
	const float **prediction, const std::size_t &prediction_rows, const std::size_t &prediction_cols, const std::size_t &prediction_stride,
	float *** hypothesis_map, const std::size_t &hypothesis_map_rows, const std::size_t &hypothesis_map_cols, const std::size_t &hypothesis_map_stride,
	float ** location_probability, const std::size_t &location_probability_rows, const std::size_t &location_probability_cols, const std::size_t &location_probability_stride,
	float ** predicted_position, const std::size_t &predicted_position_rows, const std::size_t &predicted_position_cols, const std::size_t &predicted_position_stride) // batch_size * 2

{
	const auto minus_sigma2_inv = -1.0/(sigma * sigma);

	{
		dim3 grid, block;
		TRN::GPU::Context::get_block_dims(location_hypothesis_kernel, { rows * cols, place_cells_number, batch_size }, grid, block);

		location_hypothesis_kernel << <grid, block, 0, stream >> > (
				batch_size, place_cells_number, rows, cols, minus_sigma2_inv,
				firing_rate_map, firing_rate_map_stride,
				prediction, prediction_stride,
				hypothesis_map, hypothesis_map_stride,
				scale, scale_stride);
		checkCudaErrors(cudaGetLastError());
	}

	std::size_t place_cells_number_range = place_cells_number / 2;
	const std::size_t place_cells_number_remaining = place_cells_number - place_cells_number_range * 2;

	{
		dim3 grid, block;
		TRN::GPU::Context::get_block_dims(weighted_sum_inplace_kernel, {  rows * cols, place_cells_number_range, batch_size }, grid, block);

		weighted_sum_inplace_kernel << <grid, block, 0, stream >> > (
			batch_size, place_cells_number_range, rows, cols,
			(const float **)scale, scale_stride,
			hypothesis_map, hypothesis_map_stride);
		checkCudaErrors(cudaGetLastError());
	}

	while (place_cells_number_range >= 2)
	{
		place_cells_number_range /= 2;

		dim3 grid, block;
		TRN::GPU::Context::get_block_dims(sum_inplace_kernel, { rows * cols, place_cells_number_range, batch_size }, grid, block);
		sum_inplace_kernel << <grid, block, 0, stream >> > (
			batch_size, place_cells_number_range, rows, cols,
			hypothesis_map,  hypothesis_map_stride);
		checkCudaErrors(cudaGetLastError());
	}

	assert(place_cells_number_range == 1);
	{
		dim3 grid, block;
		TRN::GPU::Context::get_block_dims(sum_kernel, { rows * cols, place_cells_number_range, batch_size }, grid, block);

		sum_kernel << <grid, block, 0, stream >> > (
			batch_size, place_cells_number_range, rows, cols,
			(const float ***)hypothesis_map, hypothesis_map_stride, 
			location_probability, location_probability_stride  );
		checkCudaErrors(cudaGetLastError());
	}


	{
		if (place_cells_number_remaining > 0)
		{
			dim3 grid, block;
			TRN::GPU::Context::get_block_dims(weighted_acc_inplace_kernel, { rows * cols, place_cells_number_range, batch_size }, grid, block);

			weighted_acc_inplace_kernel << <grid, block, 0, stream >> > (
				batch_size, place_cells_number_range, rows, cols,
				hypothesis_map, hypothesis_map_stride,
				scale, scale_stride,
				location_probability, location_probability_stride);
			checkCudaErrors(cudaGetLastError());
		}
	}






	
	/*std::vector<float *> batched_location_probability(batch_size);
	checkCudaErrors(cudaMemcpyAsync(batched_location_probability.data(), location_probability, sizeof(float *) * batch_size, cudaMemcpyKind::cudaMemcpyDeviceToHost, stream));

	for (std::size_t batch = 0; batch < batch_size; batch++)
	{
		int idx = -1;
		checkCudaErrors(cublasIsamax(handle, location_probability_stride * location_probability_cols, batched_location_probability[batch], 1, &idx));
		checkCudaErrors(cudaStreamSynchronize(stream));
		int row = idx % location_probability_stride;
		int col = idx / location_probability_stride;

		checkCudaErrors(cudaMemcpy2DAsync(&predicted_position[predicted_position_stride * 0 + batch], predicted_position_stride * sizeof(float), &x_grid[x_grid_stride * col], x_grid_stride * sizeof(float), 1 * sizeof(float), 1, cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));
		checkCudaErrors(cudaMemcpy2DAsync(&predicted_position[predicted_position_stride * 1 + batch], predicted_position_stride * sizeof(float), &y_grid[y_grid_stride * (rows - row + 1)], y_grid_stride * sizeof(float), 1 * sizeof(float), 1, cudaMemcpyKind::cudaMemcpyDeviceToDevice, stream));

	}*/







	//std::cout << "row " << row << ", col = " << col << std::endl;

	//debug_mat(stream, location_probability, rows, cols, location_probability_stride);*/
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
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
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
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
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
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
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
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
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
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
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
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
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
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
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
	const unsigned int *offsets, const unsigned int *durations, const std::size_t &repetitions,
	float *states_samples, const std::size_t &states_rows, const std::size_t &states_cols, const std::size_t &states_stride
	);

