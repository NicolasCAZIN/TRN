#pragma once

#include "engine_global.h"
#include "Executor.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Blocking : public TRN::Engine::Executor
		{
		public :
			Blocking();
			virtual ~Blocking();
		protected :
			virtual void join() override;

		public :
			virtual void run() override;
			virtual void run_one() override;
		public :
			static std::shared_ptr<Blocking> create();

		};
	};
};
