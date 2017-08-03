#pragma once

#include "engine_global.h"
#include "Communicator.h"
#include "Backend/Driver.h"
namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Worker 
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Worker(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::shared_ptr<TRN::Engine::Communicator> &communicator);
		public :
			virtual ~Worker();

		private :
			void receive();


		public :
			static std::shared_ptr<TRN::Engine::Worker> create(const std::shared_ptr<TRN::Backend::Driver> &driver, const std::shared_ptr<TRN::Engine::Communicator> &communicator);
		};
	};
};
