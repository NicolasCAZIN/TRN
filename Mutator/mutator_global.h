#pragma once



#ifndef BUILD_STATIC
# if defined(MUTATOR_LIB)
#  define MUTATOR_EXPORT __declspec(dllexport)
# else
#  define MUTATOR_EXPORT __declspec(dllimport)
# endif
#else
# define MUTATOR_EXPORT
#endif
