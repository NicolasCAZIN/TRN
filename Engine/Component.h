#pragma once

#include "engine_global.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Component
		{
		public:
			virtual void start() = 0;
			virtual void stop() = 0;
			virtual void synchronize() = 0;
		};
	};
};
