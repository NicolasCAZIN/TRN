#pragma once

#include "engine_global.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Executor
		{
		protected :
			class Handle;
			std::unique_ptr<Handle> handle;

		protected:
			Executor();
		public :
			virtual ~Executor();

		public :
			void post(const std::function<void(void)> &functor);
			void terminate();

		public :
			virtual void run() = 0;
		};
	};
};
