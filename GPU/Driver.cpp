#include "stdafx.h"

#include "Backend/Driver_impl.h"
#include "Driver_impl.h"
#include "Memory.h"
#include "Random.h"
#include "Algorithm.h"
#include "Helper/Logger.h"

static std::mutex mutex;
static std::map<std::size_t, int> counter;

static void increase_reference(const std::size_t &index)
{
	std::unique_lock<std::mutex> lock(mutex);

	counter[index]++;
}

static void decrease_reference(const std::size_t &index)
{
	std::unique_lock<std::mutex> lock(mutex);

	if (counter.find(index) == counter.end() || counter[index] == 0)
	{
		throw std::runtime_error("device #" + std::to_string(index) + " have not been initialized before");
	}
	counter[index]--;
	if (counter[index] == 0)
	{
	
		checkCudaErrors(cudaDeviceReset());
		INFORMATION_LOGGER <<   "device #" << index + 1 << " reset" ;
	}
}

TRN::GPU::Driver::Driver(const int &device) :
	TRN::GPU::Driver::Driver(TRN::GPU::Context::create(device))
{
	increase_reference(device);
}

TRN::GPU::Driver::Driver(const std::shared_ptr<TRN::GPU::Context> context) :
	handle(std::make_unique<Handle>()),
	TRN::Backend::Driver(TRN::GPU::Memory::create(context), TRN::GPU::Random::create(context), TRN::GPU::Algorithm::create(context))
{
	handle->context = context;
}
TRN::GPU::Driver::~Driver()
{
	
	handle->context.reset();
	handle.reset();
}

void TRN::GPU::Driver::dispose()
{
	handle->context->toggle();
	handle->context->dispose();
	decrease_reference(handle->context->get_device());
}

void TRN::GPU::Driver::synchronize()
{
	for (std::size_t k = 0; k < TRN::GPU::Context::STREAM_NUMBER; k++)
		checkCudaErrors(cudaStreamSynchronize(handle->context->get_streams()[k]));
}

void TRN::GPU::Driver::toggle()
{
	handle->context->toggle();
}

std::string TRN::GPU::Driver::name()
{
	return handle->context->get_name();
}
int TRN::GPU::Driver::index()
{
	return handle->context->get_device() + 1;
}

std::shared_ptr<TRN::GPU::Driver> TRN::GPU::Driver::create(const int &device)
{
	return std::make_shared<TRN::GPU::Driver>( device);
}

static std::list<std::pair<int, std::string>> enumerated;

std::list<std::pair<int, std::string>> TRN::GPU::enumerate_devices()
{
	if (enumerated.empty())
	{
		int count;
		switch (cudaGetDeviceCount(&count))
		{
			case cudaErrorInsufficientDriver :
				ERROR_LOGGER << "CUDA enabled device present but the actual driver is insufficient (version " << CUDA_VERSION << " expected)" ;
				break;
			case cudaErrorNoDevice:
			
				ERROR_LOGGER << "No CUDA enabled device had been detected" ;
				break;
			default :
				for (int k = 0; k < count; k++)
				{
					cudaDeviceProp prop;

					checkCudaErrors(cudaSetDevice(k));
					checkCudaErrors(cudaGetDeviceProperties(&prop, k));

					auto device_number = k + 1;
					auto name = prop.name;// +std::string(" #") + std::to_string(device_number);
					enumerated.push_back(std::make_pair(device_number, name));
				}
				break;
		}
	}
	return enumerated;
}