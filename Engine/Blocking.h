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
		public :
			virtual void run() override;

		public :
			static std::shared_ptr<Blocking> create();

		};
	};
};
