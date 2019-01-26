#pragma once



#ifndef BUILD_STATIC
# if defined(RESERVOIR_LIB)
#  define RESERVOIR_EXPORT __declspec(dllexport)
# else
#  define RESERVOIR_EXPORT __declspec(dllimport)
# endif
#else
# define RESERVOIR_EXPORT
#endif
