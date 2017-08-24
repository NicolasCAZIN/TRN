#include "stdafx.h"
#include "Context_impl.h"

const std::size_t TRN::GPU::Context::DEFAULT_DIV = 1;
const std::size_t TRN::GPU::Context::DEFAULT_DIMS = 1;
const std::size_t TRN::GPU::Context::DEFAULT_DYNAMIC_MEMORY_SIZE = 0;



TRN::GPU::Context::Context(const int &device) :
	handle(std::make_unique<TRN::GPU::Context::Handle>())
{
	int count;
	checkCudaErrors(cudaGetDeviceCount(&count));
	if (device >= count)
	{
		throw std::invalid_argument("GPU #" + std::to_string(device) + " is not available");
	}

	handle->device = device;
	cudaDeviceProp prop;



	checkCudaErrors(cudaSetDevice(device));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	
	if (prop.kernelExecTimeoutEnabled > 0)
	{
		throw std::runtime_error("CUDA kernel timeout must not be enabled because of persistent kernels");
	}
	//checkCudaErrors(cudaStreamCreate(&handle->stream));
	checkCudaErrors(cudaStreamCreateWithFlags(&handle->stream, cudaStreamNonBlocking));
	checkCudaErrors(cublasCreate(&handle->handle));
	checkCudaErrors(cublasSetStream(handle->handle, handle->stream));
	checkCudaErrors(cublasSetAtomicsMode(handle->handle, cublasAtomicsMode_t::CUBLAS_ATOMICS_ALLOWED));
	checkCudaErrors(curandCreateGenerator(&handle->generator, CURAND_RNG_PSEUDO_DEFAULT));
	checkCudaErrors(curandSetStream(handle->generator, handle->stream));


	
	//nppSetStream(handle->stream);
	handle->stride_alignement = prop.warpSize;
	handle->name.assign(prop.name);
	handle->max_block_size = prop.maxThreadsPerBlock;


	std::cout << "GPU version selected : " << prop.name << " # " << (device + 1) << std::endl;
	
}

TRN::GPU::Context::~Context()
{
	checkCudaErrors(cudaSetDevice(handle->device));
	checkCudaErrors(cudaStreamSynchronize(handle->stream));
	checkCudaErrors(cudaStreamDestroy(handle->stream));
	checkCudaErrors(cublasDestroy(handle->handle));
	checkCudaErrors(curandDestroyGenerator(handle->generator));

	handle.reset();
}

void TRN::GPU::Context::toggle()
{

	checkCudaErrors(cudaSetDevice(handle->device));
}

const std::size_t &TRN::GPU::Context::get_stride_alignment()
{
	return handle->stride_alignement;
}

const int &TRN::GPU::Context::get_device()
{
	return handle->device;
}
const cudaStream_t &TRN::GPU::Context::get_stream()
{
	return handle->stream;
}
const curandGenerator_t &TRN::GPU::Context::get_generator()
{
	return handle->generator;
}
const cublasHandle_t &TRN::GPU::Context::get_handle()
{
	return handle->handle;
}

const std::string &TRN::GPU::Context::get_name()
{
	return handle->name;
}


std::shared_ptr<TRN::GPU::Context> TRN::GPU::Context::create(const int &device)
{
	return std::make_shared<TRN::GPU::Context>(device);
}