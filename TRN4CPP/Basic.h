#pragma once

#include "trn4cpp_global.h"

namespace TRN4CPP
{
	namespace Logging
	{
		namespace Severity
		{
			namespace Trace
			{
				void TRN4CPP_EXPORT setup();
			}
			namespace Debug
			{
				void TRN4CPP_EXPORT setup();
			}
			namespace Information
			{
				void TRN4CPP_EXPORT setup();
			}
			namespace Warning
			{
				void TRN4CPP_EXPORT setup();
			}
			namespace Error
			{
				void TRN4CPP_EXPORT setup();
			}
		}
	};

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
	};

	namespace Simulation
	{
		void TRN4CPP_EXPORT encode(const unsigned short &frontend, const unsigned short &condition_number, const unsigned int &simulation_number, unsigned long long &id);
		void TRN4CPP_EXPORT decode(const unsigned long long &id, unsigned short &frontend, unsigned short &condition_number, unsigned int &simulation_number);
		
		extern TRN4CPP_EXPORT const std::string DEFAULT_TAG;
	};
};


