#pragma once



#ifndef BUILD_STATIC
# if defined(MATFILE_LIB)
#  define MATFILE_EXPORT __declspec(dllexport)
# else
#  define MATFILE_EXPORT __declspec(dllimport)
# endif
#else
# define MATFILE_EXPORT
#endif
