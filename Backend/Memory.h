#pragma once

#include "backend_global.h"

namespace TRN
{
	namespace Backend
	{
		class BACKEND_EXPORT Memory
		{
		protected:
			static const std::size_t DEFAULT_HEIGHT;
			static const bool DEFAULT_ASYNC;

		public :
			virtual ~Memory();

		public:
			virtual void align(const std::size_t &unaligned, std::size_t &aligned) = 0;
			virtual void allocate(void **block, const std::size_t &depth, const std::size_t &size) = 0;
			virtual void allocate(void **block, std::size_t &stride, const std::size_t &depth, const std::size_t &width, const std::size_t &height = DEFAULT_HEIGHT) = 0;
			virtual void deallocate(void *remote) = 0;
			virtual void copy(const void *src, void *dst, const std::size_t &depth, const std::size_t &size, const bool &async = DEFAULT_ASYNC) = 0;
			virtual void copy(const void *src, void *dst, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &src_stride, const std::size_t &dst_stride, const bool &async = DEFAULT_ASYNC) = 0;
			virtual void upload(const void *local, void *remote, const std::size_t &depth, const std::size_t &size, const bool &async = DEFAULT_ASYNC) = 0;
			virtual void upload(const void *local, void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async = DEFAULT_ASYNC) = 0;
			virtual void download(void *local, const void *remote, const std::size_t &depth, const std::size_t &size, const bool &async = DEFAULT_ASYNC) = 0;
			virtual void download(void *local, const void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async = DEFAULT_ASYNC) = 0;
			virtual void blank(void *remote, const std::size_t &depth, const std::size_t &width, const std::size_t &height, const std::size_t &stride, const bool &async = DEFAULT_ASYNC) = 0;
		
		};
	};
};
