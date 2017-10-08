#pragma once



#ifndef TRN_STATIC
# if defined(BACKEND_LIB)
#  define BACKEND_EXPORT __declspec(dllexport)
# else
#  define BACKEND_EXPORT __declspec(dllimport)
# endif
#else
# define BACKEND_EXPORT
#endif
