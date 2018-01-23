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
			void start();
			void dispose();
			void terminate();
		public:
			void update_processor(const int &rank, const std::string host, const unsigned int &index, const std::string name);
			//void wait_not_allocated();
			std::shared_ptr<TRN::Engine::Processor> allocate(const unsigned long long &simulation_id);
			void deallocate(const unsigned long long &simulation_id);
		
			std::shared_ptr<TRN::Engine::Processor> retrieve(const unsigned long long &simulation_id);

			std::vector<std::shared_ptr<TRN::Engine::Processor>> get_processors();

		public :
			static std::shared_ptr<TRN::Engine::Manager> create(const std::size_t &size);
		};

	};
};
	