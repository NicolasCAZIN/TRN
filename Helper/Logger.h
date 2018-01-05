#pragma once

#include "helper_global.h"
#include <omp.h>

namespace TRN
{
	namespace Helper
	{


		class   Logger : public std::ostringstream
		{
		public:
			enum  Severity
			{
				TRACE_LEVEL = 0,
				DEBUG_LEVEL,
				INFORMATION_LEVEL,
				WARNING_LEVEL,
				ERROR_LEVEL
			};
		public:
			HELPER_EXPORT Logger( const TRN::Helper::Logger::Severity  &severity, const std::string &module);
			HELPER_EXPORT ~Logger();

		private :
			class Handle;

			std::unique_ptr<Handle> handle;
		
		public :
			static void HELPER_EXPORT setup(const TRN::Helper::Logger::Severity  &severity, const bool &exit_on_error = true);
		};
	}
}

#define LOCATION " (LOCATION Line : " << __LINE__ << ", File : " << __FILE__ <<  ", Function : " << __FUNCTION__  << ") "
#define OMP "(OpenMP processors available : " << omp_get_num_procs() << ", thread " << omp_get_thread_num()  << "/" << omp_get_num_threads() <<"[" << omp_get_max_threads() << "]" << ") "
#define TRACE_LOGGER TRN::Helper::Logger{TRN::Helper::Logger::TRACE_LEVEL, TRN_MODULE} << __FUNCTION__
#define DEBUG_LOGGER  TRN::Helper::Logger{TRN::Helper::Logger::DEBUG_LEVEL, TRN_MODULE} << OMP
#define INFORMATION_LOGGER  TRN::Helper::Logger{TRN::Helper::Logger::INFORMATION_LEVEL, TRN_MODULE}
#define WARNING_LOGGER  TRN::Helper::Logger{TRN::Helper::Logger::WARNING_LEVEL, TRN_MODULE} << LOCATION
#define ERROR_LOGGER  TRN::Helper::Logger{TRN::Helper::Logger::ERROR_LEVEL, TRN_MODULE} << LOCATION

namespace std
{
	HELPER_EXPORT std::istream &operator >> (std::istream &is, TRN::Helper::Logger::Severity &severity);
}

/*BOOST_LOG_GLOBAL_LOGGER(sysLogger, boost::log::sources::severity_channel_logger_mt<boost::log::trivial::severity_level>);
BOOST_LOG_GLOBAL_LOGGER(dataLogger, boost::log::sources::severity_channel_logger_mt<boost::log::trivial::severity_level>);

namespace TRN
{
	namespace Helper
	{
		class HELPER_EXPORT Logger
		{
		public:
			static void initialize(const std::string& configFileName = "");

			/// Disable logging
			static void disable();

			/// Add a file sink for LOG_DATA_* for >= INFO.
			/// This file sink will be used along with any configured via Config in init().
			static void addDataFileLog(const std::string& logFileName);
		};
	};
};

#define LOG_LOG_LOCATION(LOGGER, LEVEL, ARG)            \
  BOOST_LOG_SEV(LOGGER, boost::log::trivial::LEVEL)     \
    << boost::log::add_value("Line", __LINE__)          \
    << boost::log::add_value("File", __FILE__)          \
    << boost::log::add_value("Function", __FUNCTION__) << ARG;

/// System Log macros.
/// TRACE < DEBUG < INFO < WARN < ERROR < FATAL
#define LOG_TRACE(ARG) LOG_LOG_LOCATION(sysLogger::get(), trace, ARG);
#define LOG_DEBUG(ARG) LOG_LOG_LOCATION(sysLogger::get(), debug, ARG);
#define LOG_INFO(ARG)  LOG_LOG_LOCATION(sysLogger::get(), info, ARG);
#define LOG_WARN(ARG)  LOG_LOG_LOCATION(sysLogger::get(), warning, ARG);
#define LOG_ERROR(ARG) LOG_LOG_LOCATION(sysLogger::get(), error, ARG);
#define LOG_FATAL(ARG) LOG_LOG_LOCATION(sysLogger::get(), fatal, ARG);

/// Data Log macros. Does not include LINE, FILE, FUNCTION.
/// TRACE < DEBUG < INFO < WARN < ERROR < FATAL
#define LOG_DATA_TRACE(ARG) BOOST_LOG_SEV(dataLogger::get(), boost::log::trivial::trace) << ARG
#define LOG_DATA_DEBUG(ARG) BOOST_LOG_SEV(dataLogger::get(), boost::log::trivial::debug) << ARG
#define LOG_DATA_INFO(ARG)  BOOST_LOG_SEV(dataLogger::get(), boost::log::trivial::info) << ARG
#define LOG_DATA_WARN(ARG)  BOOST_LOG_SEV(dataLogger::get(), boost::log::trivial::warning) << ARG
#define LOG_DATA_ERROR(ARG) BOOST_LOG_SEV(dataLogger::get(), boost::log::trivial::error) << ARG
#define LOG_DATA_FATAL(ARG) BOOST_LOG_SEV(dataLogger::get(), boost::log::trivial::fatal) << ARG*/
