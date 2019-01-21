#pragma once

#include "Component.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Task : public TRN::Engine::Component
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		protected :
			Task();

		public :
			virtual ~Task();

		public :
			 virtual void start() override;
			 virtual void stop() override;
		protected:
			void cancel();

		protected :
			virtual void initialize();
			virtual void body() = 0;
			virtual void uninitialize();
			virtual void joined();
	
		private:
			bool stop_requested();
			
		};
	}
};