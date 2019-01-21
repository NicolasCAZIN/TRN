#include "stdafx.h"
#include "Context_impl.h"
#include "Helper/Logger.h"
const std::size_t TRN::GPU::Context::STREAM_NUMBER = 128;
const std::size_t TRN::GPU::Context::EVENT_NUMBER = STREAM_NUMBER;
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
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceBlockingSync));
	checkCudaErrors(cudaGetDeviceProperties(&prop, device));
	auto freq_ghz = prop.clockRate *1e-6f;
	/*if (prop.kernelExecTimeoutEnabled > 0)
	{
		throw std::runtime_error("CUDA kernel timeout must not be enabled because of persistent kernels");
	}*/
	//checkCudaErrors(cudaStreamCreate(&handle->stream));

	handle->streams.resize(STREAM_NUMBER);
	handle->handles.resize(STREAM_NUMBER);
	handle->events.resize(EVENT_NUMBER);
	for (std::size_t k = 0; k < EVENT_NUMBER; k++)
	{
		checkCudaErrors(cudaEventCreateWithFlags(&handle->events[k], cudaEventDisableTiming));
	}
	for (std::size_t k = 0; k < STREAM_NUMBER; k++)
	{
		checkCudaErrors(cudaStreamCreateWithFlags(&handle->streams[k], cudaStreamNonBlocking));
		checkCudaErrors(cublasCreate(&handle->handles[k]));
		checkCudaErrors(cublasSetStream(handle->handles[k], handle->streams[k]));
	
		checkCudaErrors(cublasSetPointerMode(handle->handles[k], cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE));
		checkCudaErrors(cublasSetMathMode(handle->handles[k], cublasMath_t::CUBLAS_TENSOR_OP_MATH));
		checkCudaErrors(cublasSetAtomicsMode(handle->handles[k], cublasAtomicsMode_t::CUBLAS_ATOMICS_ALLOWED));

	}

	auto simd_cores = _ConvertSMVer2Cores(prop.major, prop.minor) * prop.multiProcessorCount;
	auto gflops = 2 * simd_cores * freq_ghz;
	
	std::stringstream stream;
	stream << "NVidia(R) " << prop.name << " GPU @ " << std::fixed << std::setprecision(2) << freq_ghz << "GHz / " << simd_cores << " simd cores (" << gflops << " GFLOPS/s)";
	std::string name = stream.str();

	//nppSetStream(handle->stream);
	handle->stride_alignement = prop.warpSize;
	handle->name = name;
	handle->max_block_size = prop.maxThreadsPerBlock;



	DEBUG_LOGGER << "Max threads per block : " << handle->max_block_size;
	INFORMATION_LOGGER <<   "GPU version selected : " << name << " # " << (device + 1) ;
	if (handle->max_block_size != 32 * 32)
	{
		ERROR_LOGGER << "maxThreadsPerBlock must be 32 * 32. Aborting";
		abort();
	}
}

void TRN::GPU::Context::dispose()
{

	//checkCudaErrors(curandDestroyGenerator(handle->generator));
}

TRN::GPU::Context::~Context()
{
	for (std::size_t k = 0; k < STREAM_NUMBER; k++)
	{
		checkCudaErrors(cudaStreamSynchronize(handle->streams[k]));
		checkCudaErrors(cublasDestroy(handle->handles[k]));
		checkCudaErrors(cudaStreamDestroy(handle->streams[k]));
	}
	for (std::size_t k = 0; k < EVENT_NUMBER; k++)
	{
		checkCudaErrors(cudaEventDestroy(handle->events[k]));
	}
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
const cudaStream_t *TRN::GPU::Context::get_streams()
{
	return handle->streams.data();
}
/*const curandGenerator_t &TRN::GPU::Context::get_generator()
{
	return handle->generator;
}*/
const cublasHandle_t *TRN::GPU::Context::get_handles()
{
	return handle->handles.data();
}
const cudaEvent_t *TRN::GPU::Context::get_events()
{
	return handle->events.data();
}
const std::string &TRN::GPU::Context::get_name()
{
	return handle->name;
}


std::shared_ptr<TRN::GPU::Context> TRN::GPU::Context::create(const int &device)
{
	return std::make_shared<TRN::GPU::Context>(device);
}