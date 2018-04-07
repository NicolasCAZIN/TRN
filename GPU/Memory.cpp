#include "stdafx.h"
#include "Memory_impl.h"
#include "Driver.cuh"

template class GPU_EXPORT std::unique_ptr<TRN::GPU::Memory::Handle>;

TRN::GPU::Memory::Memory(const std::shared_ptr<TRN::GPU::Context> context) : handle(std::make_unique<TRN::GPU::Memory::Handle>(context))
{

}

TRN::GPU::Memory::~Memory()
{
	handle.reset();
}

void TRN::GPU::Memory::align(const std::size_t &unaligned, std::size_t &aligned)
{
	const std::size_t reminder = unaligned % handle->context->get_stride_alignment();
	aligned = unaligned + (reminder ? (handle->context->get_stride_alignment() - reminder) : (0));
}


void TRN::GPU::Memory::allocate(void **block, const std::size_t &depth, const std::size_t &size)
{
	checkCudaErrors(cudaMalloc(block, size * depth));
}
void TRN::GPU::Memory::allocate(void **block, std::size_t &stride, const std::size_t &depth, const std::size_t &width, const std::size_t &height)
{
	std::size_t pitch;

	checkCudaErrors(cudaMallocPitch(block, &pitch,width * depth, height));
	stride = pitch / depth;

}
void  TRN::GPU::Memory::deallocate(void *remote)
{
	checkCudaErrors(cudaFree(remote));
}


void  TRN::GPU::Memory::copy(const void *src, void *dst, const std::size_t &depth, const std::size_t &size, const bool &async)
{
	checkCudaErrors(cudaMemcpyAsync(dst, src, size * depth, cudaMemcpyKind::cudaMemcpyDeviceToDevice, *handle->context->get_streams()));
	if (!async)
		checkCudaErrors(cudaStreamSynchronize(*handle->context->get_streams()));
}

void  TRN::GPU::Memory::copy(const void *src, void *dst, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &src_stride, const std::size_t &dst_stride, const bool &async)
{
	checkCudaErrors(cudaMemcpy2DAsync(dst, dst_stride * depth, src, src_stride * depth,width * depth, height, cudaMemcpyKind::cudaMemcpyDeviceToDevice, *handle->context->get_streams()));
	if (!async)
		checkCudaErrors(cudaStreamSynchronize(*handle->context->get_streams()));
}
void  TRN::GPU::Memory::upload(const void *local, void *remote, const std::size_t &depth, const std::size_t &size, const bool &async)
{
	checkCudaErrors(cudaMemcpyAsync(remote, local, depth * size, cudaMemcpyKind::cudaMemcpyHostToDevice, *handle->context->get_streams()));
	if (!async)
		checkCudaErrors(cudaStreamSynchronize(*handle->context->get_streams()));
}
void  TRN::GPU::Memory::upload(const void *local, void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async)
{
	checkCudaErrors(cudaMemcpy2DAsync(remote, stride * depth, local,width * depth,width * depth, height, cudaMemcpyKind::cudaMemcpyHostToDevice, *handle->context->get_streams()));
	if (!async)
		checkCudaErrors(cudaStreamSynchronize(*handle->context->get_streams()));
}

void  TRN::GPU::Memory::download(void *local, const void *remote, const std::size_t &depth, const std::size_t &size, const bool &async)
{
	checkCudaErrors(cudaMemcpyAsync(local, remote, size * depth, cudaMemcpyKind::cudaMemcpyDeviceToHost, *handle->context->get_streams()));
	if (!async)
		checkCudaErrors(cudaStreamSynchronize(*handle->context->get_streams()));
}
void  TRN::GPU::Memory::download(void *local, const void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async)
{
	checkCudaErrors(cudaMemcpy2DAsync(local,width * depth, remote, stride * depth,width * depth, height, cudaMemcpyKind::cudaMemcpyDeviceToHost, *handle->context->get_streams()));
	if (!async)
		checkCudaErrors(cudaStreamSynchronize(*handle->context->get_streams()));
}

void  TRN::GPU::Memory::blank(void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async)
{
	checkCudaErrors(cudaMemset2DAsync(remote, stride * sizeof(float), 0,width * sizeof(float), height, *handle->context->get_streams()));
	if (!async)
		checkCudaErrors(cudaStreamSynchronize(*handle->context->get_streams()));
}


std::shared_ptr<TRN::GPU::Memory> TRN::GPU::Memory::create(const std::shared_ptr<TRN::GPU::Context> context)
{
	return std::make_shared<TRN::GPU::Memory>(context);
}