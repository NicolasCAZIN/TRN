#pragma once

#include "gpu_global.h"

#include "Backend/Memory.h"
#include "Context.h"

namespace TRN
{
	namespace GPU
	{
		class  Memory : public TRN::Backend::Memory
		{
		private :
			class Handle;
			 std::unique_ptr<Handle>  handle;

		public :
			Memory(const std::shared_ptr<Context> context);
			~Memory();

		public :
			virtual void GPU_EXPORT align(const std::size_t &unaligned, std::size_t &aligned) override;
			virtual void GPU_EXPORT allocate(void **block, const std::size_t &depth, const std::size_t &size) override;
			virtual void GPU_EXPORT allocate(void **block, std::size_t &stride, const std::size_t &depth, const std::size_t &width, const std::size_t &height) override;
			virtual void GPU_EXPORT deallocate(void *remote) override;
			virtual void GPU_EXPORT copy(const void *src, void *dst, const std::size_t &depth, const std::size_t &size, const bool &async) override;
			virtual void GPU_EXPORT copy(const void *src, void *dst, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &src_stride, const std::size_t &dst_stride, const bool &async) override;
			virtual void GPU_EXPORT upload(const void *local, void *remote, const std::size_t &depth, const std::size_t &size, const bool &async) override;
			virtual void GPU_EXPORT upload(const void *local, void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async) override;
			virtual void GPU_EXPORT download(void *local, const void *remote, const std::size_t &depth, const std::size_t &size, const bool &async) override;
			virtual void GPU_EXPORT download(void *local, const void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async) override;
			virtual void GPU_EXPORT blank(void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async) override;

		public :
			static std::shared_ptr<Memory> GPU_EXPORT  create(const std::shared_ptr<Context> context);
		};
	};
};
