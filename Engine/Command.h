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

		public:
			Executor();
			~Executor();

		public :
			void post(const std::function<void(void)> &functor);
		public:
	
		public :
			void join();

			static std::shared_ptr<Executor> create();
		};
	};
};
