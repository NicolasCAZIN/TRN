#pragma once

#include "engine_global.h"
#include "Processor.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Manager
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Manager(const std::size_t &size);

		public :
			virtual ~Manager();

		public:
			void wait_not_allocated();
			std::shared_ptr<TRN::Engine::Processor> allocate(const unsigned int &id);
			void deallocate(const unsigned int &id);
		
			std::shared_ptr<TRN::Engine::Processor> retrieve(const unsigned int &id);

			std::vector<std::shared_ptr<TRN::Engine::Processor>> get_processors();

		public :
			static std::shared_ptr<TRN::Engine::Manager> create(const std::size_t &size);
		};

	};
};
	