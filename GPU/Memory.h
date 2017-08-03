#pragma once

#include "gpu_global.h"

#include "Backend/Memory.h"
#include "Context.h"

namespace TRN
{
	namespace GPU
	{
		class GPU_EXPORT Memory : public TRN::Backend::Memory
		{
		private :
			class Handle;
			mutable std::unique_ptr<Handle> handle;

		public :
			Memory(const std::shared_ptr<Context> context);
			~Memory();

		public :
			virtual void align(const std::size_t &unaligned, std::size_t &aligned) override;
			virtual void allocate(void **block, const std::size_t &depth, const std::size_t &size) override;
			virtual void allocate(void **block, std::size_t &stride, const std::size_t &depth, const std::size_t &width, const std::size_t &height) override;
			virtual void deallocate(void *remote) override;
			virtual void copy(const void *src, void *dst, const std::size_t &depth, const std::size_t &size, const bool &async) override;
			virtual void copy(const void *src, void *dst, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &src_stride, const std::size_t &dst_stride, const bool &async) override;
			virtual void upload(const void *local, void *remote, const std::size_t &depth, const std::size_t &size, const bool &async) override;
			virtual void upload(const void *local, void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async) override;
			virtual void download(void *local, const void *remote, const std::size_t &depth, const std::size_t &size, const bool &async) override;
			virtual void download(void *local, const void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async) override;
			virtual void blank(void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async) override;
			virtual bool is_column_major() override;

		public :
			static std::shared_ptr<Memory> create(const std::shared_ptr<Context> context);
		};
	};
};
