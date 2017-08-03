#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <arrayfire.h>
#include <af/cuda.h>
#include <helper_cuda.h>
#include <ctime>
int main(int argc, char *argv[])
{
	int batch_size = 10;
	int epochs = 1000;
	int observations = 130;
	int reservoir_size = 1024;
	int stimulus_size = 256;
	int prediction_size = stimulus_size;

	int device = 3;

	af::setBackend(af::Backend::AF_BACKEND_CUDA);
	af::setDevice(device);

	cublasHandle_t handle;
	checkCudaErrors(cublasCreate(&handle));
	cublasSetStream(handle, afcu::getStream(device));
	std::vector<af::array> stimulus(batch_size);
	std::vector<af::array> w_ffwd(batch_size);
	std::vector<af::array> u_ffwd(batch_size);

	thrust::device_vector<float *> batched_stimulus(batch_size);
	thrust::device_vector<float *> batched_w_ffwd(batch_size);
	thrust::device_vector<float *> batched_u_ffwd(batch_size);

	for (std::size_t k = 0; k < batch_size; k++)
	{
		stimulus[k] = af::array(observations, stimulus_size);
		u_ffwd[k] = af::array(observations, reservoir_size);
		w_ffwd[k] = af::randu(reservoir_size, stimulus_size);

		batched_stimulus[k] = stimulus[k].device<float>();
		batched_w_ffwd[k] = w_ffwd[k].device<float>();
		batched_u_ffwd[k] = u_ffwd[k].device<float>();

	}
	const float one = 1.0f;
	const float zero = 0.0f;

	auto t0 = std::clock();
	auto op_a_rows = observations;
	auto op_a_cols = stimulus_size;
	auto op_b_rows = stimulus_size;
	auto op_b_cols = reservoir_size;


	auto m = op_a_rows;

	auto n = op_b_cols;

	auto k = op_a_cols;
	checkCudaErrors(cublasSgemmBatched(
		handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T,
		m, n, k, &one,
		(const float **)thrust::raw_pointer_cast(batched_stimulus.data()),
		observations,
		(const float **)thrust::raw_pointer_cast(batched_w_ffwd.data()),
		reservoir_size,
		&zero,
		thrust::raw_pointer_cast(batched_u_ffwd.data()),
		observations,
		batch_size
	));
	for (int epoch = 0; epoch < epochs; epochs++)
	{

		for (int t = 0; t < observations; t++)
		{

		}

	}

	auto t1 = std::clock();
	auto cycles = observations * epochs * batch_size;

	checkCudaErrors(cublasDestroy(handle));

	return 0;
}