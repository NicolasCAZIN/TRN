#pragma once

#include "Task.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Executor : public TRN::Engine::Task
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Executor();
			virtual ~Executor();

		protected :
			virtual void body() override;

		public :
			void terminate();
			void post(const std::function<void(void)> &command);

		public :
			static std::shared_ptr<Executor> create();
		};
	};
};
