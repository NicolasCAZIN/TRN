#pragma once



#ifndef BUILD_STATIC
# if defined(SIMULATOR_LIB)
#  define SIMULATOR_EXPORT __declspec(dllexport)
# else
#  define SIMULATOR_EXPORT __declspec(dllimport)
# endif
#else
# define SIMULATOR_EXPORT
#endif
