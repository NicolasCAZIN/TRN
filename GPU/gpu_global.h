#pragma once



#ifndef TRN_STATIC
# if defined(GPU_LIB)
#  define GPU_EXPORT __declspec(dllexport)
# else
#  define GPU_EXPORT __declspec(dllimport)
# endif
#else
# define GPU_EXPORT
#endif
