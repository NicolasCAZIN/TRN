#pragma once

#include "engine_global.h"
#include "Executor.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT NonBlocking : public TRN::Engine::Executor
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			NonBlocking();
			virtual ~NonBlocking();

		protected :
			virtual void join() override;
		public :
			virtual void run() override;
			virtual void run_one() override;
		public :
			static std::shared_ptr<NonBlocking> create();
		};
	};
};
