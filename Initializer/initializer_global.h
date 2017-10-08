#pragma once



#ifndef TRN_STATIC
# if defined(INITIALIZER_LIB)
#  define INITIALIZER_EXPORT __declspec(dllexport)
# else
#  define INITIALIZER_EXPORT __declspec(dllimport)
# endif
#else
# define INITIALIZER_EXPORT
#endif
