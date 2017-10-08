#pragma once



#ifndef BUILD_STATIC
# if defined(MEASUREMENT_LIB)
#  define MEASUREMENT_EXPORT __declspec(dllexport)
# else
#  define MEASUREMENT_EXPORT __declspec(dllimport)
# endif
#else
# define MEASUREMENT_EXPORT
#endif
