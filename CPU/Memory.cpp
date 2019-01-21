#include "stdafx.h"
#include "Memory.h"

static void *row_ptr(const void *ptr, const std::size_t &depth, const std::size_t &stride, const std::size_t &row)
{
	return (void *)((char *)ptr + stride*depth*row);
}

template<TRN::CPU::Implementation Implementation>
static const std::size_t aligned_size(const std::size_t &unaligned)
{
	const std::size_t reminder = unaligned % (TRN::CPU::Traits<Implementation>::step * sizeof(float));
	if (reminder)
		return unaligned + (TRN::CPU::Traits<Implementation>::step * sizeof(float) - reminder);
	else
		return unaligned;
}

template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Memory<Implementation>::allocate_implementation(void **block, std::size_t &stride, const std::size_t &depth, const std::size_t &width, const std::size_t &height)
{
	assert(block != NULL);
	stride = aligned_size<Implementation>(width * depth)/depth;
	auto horizontal_stride = aligned_size<Implementation>(height * depth) / depth;
	*block = _mm_malloc(horizontal_stride * stride * depth, TRN::CPU::Traits<Implementation>::step * sizeof(float));
}
template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Memory<Implementation>::deallocate_implementation(void *remote)
{
	_mm_free(remote);
}
template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Memory<Implementation>::copy_implementation(const void *src, void *dst, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &src_stride, const std::size_t &dst_stride, const bool &async)
{
//#pragma omp parallel for
	for (int row = 0; row < height; row++)
	{
		std::memcpy(row_ptr(dst, depth, dst_stride, row), row_ptr(src, depth, src_stride, row),width * depth);
	}
}
template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Memory<Implementation>::blank_implementation(void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride)
{
	std::memset(remote, 0, height * stride * depth);
}

template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Memory<Implementation>::align(const std::size_t &unaligned, std::size_t &aligned)
{
	aligned = aligned_size<Implementation>(unaligned);
}

template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Memory<Implementation>::allocate(void **block, std::size_t &stride, const std::size_t &depth, const std::size_t &width, const std::size_t &height)
{
	TRN::CPU::Memory<Implementation>::allocate_implementation(block, stride, depth,width, height);
}
template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Memory<Implementation>::allocate(void **block, const std::size_t &depth, const std::size_t &size)
{
	*block = _mm_malloc(depth * size, TRN::CPU::Traits<Implementation>::step * sizeof(float));
}

template<TRN::CPU::Implementation Implementation>
void  TRN::CPU::Memory<Implementation>::deallocate(void *remote)
{
	TRN::CPU::Memory<Implementation>::deallocate_implementation(remote);
}
template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Memory<Implementation>::copy(const void *src, void *dst, const std::size_t &depth, const std::size_t &size, const bool &async)
{
	std::memcpy(dst, src, depth * size);
}


template<TRN::CPU::Implementation Implementation>
void  TRN::CPU::Memory<Implementation>::copy(const void *src, void *dst, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &src_stride, const std::size_t &dst_stride, const bool &async)
{
	TRN::CPU::Memory<Implementation>::copy_implementation(src, dst, depth,width, height, src_stride, dst_stride, async);
}

template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Memory<Implementation>::upload(const void *local, void *remote, const std::size_t &depth, const std::size_t &size, const bool &async)
{
	std::memcpy(remote, local, depth * size);
}

template<TRN::CPU::Implementation Implementation>
void  TRN::CPU::Memory<Implementation>::upload(const void *local, void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async)
{
//#pragma omp parallel for
	for (int row = 0; row < height; row++)
	{
		std::memcpy(row_ptr(remote, depth, stride, row), row_ptr(local, depth,width, row),width * depth);
	}
}

template<TRN::CPU::Implementation Implementation>
void TRN::CPU::Memory<Implementation>::download(void *local, const void *remote, const std::size_t &depth, const std::size_t &size, const bool &async)
{
	std::memcpy(local, remote, depth * size);
}


template<TRN::CPU::Implementation Implementation>
void  TRN::CPU::Memory<Implementation>::download(void *local, const void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async)
{
//#pragma omp parallel for
	for (int row = 0; row < height; row++)
	{
		std::memcpy(row_ptr(local, depth,width, row), row_ptr(remote, depth, stride, row),width * depth);
	}
}

template<TRN::CPU::Implementation Implementation>
void  TRN::CPU::Memory<Implementation>::blank(void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async)
{
//#pragma omp parallel for
	TRN::CPU::Memory<Implementation>::blank_implementation(remote, depth,width, height, stride);

}

template<TRN::CPU::Implementation Implementation>
std::shared_ptr<TRN::CPU::Memory<Implementation>> TRN::CPU::Memory<Implementation>::create()
{
	return std::make_shared<TRN::CPU::Memory<Implementation>>();
}
template TRN::CPU::Memory<TRN::CPU::Implementation::SCALAR>;
#if !defined(_M_IX86) && defined(_M_X64)
template TRN::CPU::Memory<TRN::CPU::Implementation::SSE2>;
template TRN::CPU::Memory<TRN::CPU::Implementation::SSE3>;
template TRN::CPU::Memory<TRN::CPU::Implementation::SSE41>;
template TRN::CPU::Memory<TRN::CPU::Implementation::AVX>;
template TRN::CPU::Memory<TRN::CPU::Implementation::AVX2_FMA3>;
#endif
#if defined(_M_IX86) && !defined(_M_X64)
template TRN::CPU::Memory<TRN::CPU::Implementation::MMX_SSE>;
#endif