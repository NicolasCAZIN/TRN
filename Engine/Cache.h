#pragma once

#include "engine_global.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Cache
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			static void initialize();
			static void uninitialize();
		private :
			static std::mutex mutex;

		public :
			Cache();
			~Cache();

		public :
			std::set<unsigned int> cached();
			void store(const unsigned int &checksum, const std::vector<float> &data);
			bool contains(const unsigned int &checksum);
			std::vector<float> retrieve(const unsigned int &checksum);

		public :
			static std::shared_ptr<Cache> create();
		};
	}
};
