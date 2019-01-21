#pragma once



#ifndef BUILD_STATIC
# if defined(MONITOR_LIB)
#  define MONITOR_EXPORT __declspec(dllexport)
# else
#  define MONITOR_EXPORT __declspec(dllimport)
# endif
#else
# define MONITOR_EXPORT
#endif
