#pragma once



#ifndef BUILD_STATIC
# if defined(RAT_LIB)
#  define RAT_EXPORT __declspec(dllexport)
# else
#  define RAT_EXPORT __declspec(dllimport)
# endif
#else
# define RAT_EXPORT
#endif
