#pragma once



#ifndef BUILD_STATIC
# if defined(TRN4MATLAB_LIB)
#  define TRN4MATLAB_EXPORT __declspec(dllexport)
# else
#  define TRN4MATLAB_EXPORT __declspec(dllimport)
# endif
#else
# define TRN4MATLAB_EXPORT
#endif
