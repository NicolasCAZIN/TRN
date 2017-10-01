#pragma once

#include "trn4cpp_global.h"

namespace TRN4CPP
{
	namespace Plugin
	{
		namespace Basic
		{
			class TRN4CPP_EXPORT Interface
			{
			public :
				virtual void initialize(const std::map<std::string, std::string>  &arguments) = 0;
				virtual void uninitialize() = 0;
			};
		}
	};

	namespace Engine
	{
		void TRN4CPP_EXPORT  	initialize();
		void TRN4CPP_EXPORT  	uninitialize();

		namespace Backend
		{
			namespace Local
			{
				void TRN4CPP_EXPORT  	initialize(const std::vector<unsigned int> &indices = {});
			};

			namespace Remote
			{
				extern TRN4CPP_EXPORT const std::string DEFAULT_HOST;
				extern TRN4CPP_EXPORT const unsigned short DEFAULT_PORT;

				void TRN4CPP_EXPORT  	initialize(const std::string &host = DEFAULT_HOST, const unsigned short &port = DEFAULT_PORT);
			};

			namespace Distributed
			{
				void TRN4CPP_EXPORT  	initialize(int argc, char *argv[]);
			};
		};
		namespace Execution
		{
			extern TRN4CPP_EXPORT const bool DEFAULT_BLOCKING;

			void TRN4CPP_EXPORT		initialize(const bool &blocking = DEFAULT_BLOCKING);
		};


	};

	namespace Simulation
	{
		extern TRN4CPP_EXPORT const std::string DEFAULT_TAG;
	};




};


