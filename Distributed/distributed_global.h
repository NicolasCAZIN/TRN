#pragma once



#ifndef BUILD_STATIC
# if defined(DISTRIBUTED_LIB)
#  define DISTRIBUTED_EXPORT __declspec(dllexport)
# else
#  define DISTRIBUTED_EXPORT __declspec(dllimport)
# endif
#else
# define DISTRIBUTED_EXPORT
#endif
