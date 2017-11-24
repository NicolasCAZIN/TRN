#pragma once

#include "engine_global.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Task
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		protected :
			Task();
		public :
			virtual ~Task() ;

		public :
			 void start();

		
			 
		protected :
			virtual void initialize();
			virtual void body() = 0;
			virtual void uninitialize();
			void join();
			void stop();
		};
	}
};