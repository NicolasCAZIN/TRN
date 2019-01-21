#pragma once
 
#ifndef BUILD_STATIC
# if defined(SIMULATEDANNEALING_LIB)
#  define SIMULATEDANNEALING_EXPORT __declspec(dllexport)
# else
#  define SIMULATEDANNEALING_EXPORT __declspec(dllimport)
# endif
#else
# define SIMULATEDANNEALING_EXPORT
#endif
