#pragma once



#ifndef BUILD_STATIC
# if defined(NETWORK_LIB)
#  define NETWORK_EXPORT __declspec(dllexport)
# else
#  define NETWORK_EXPORT __declspec(dllimport)
# endif
#else
# define NETWORK_EXPORT
#endif

