#pragma once

#include "engine_global.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Receiver
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		protected :
			Receiver();
			virtual ~Receiver();

		public :
			virtual void start();
			
		protected :
			bool is_running();
			virtual void initialize();
			virtual void receive() = 0;
			virtual void uninitialize();
			void join();
			void stop();
		};
	}
};