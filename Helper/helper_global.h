#pragma once



#ifndef TRN_STATIC
# if defined(HELPER_LIB)
#  define HELPER_EXPORT __declspec(dllexport)
# else
#  define HELPER_EXPORT __declspec(dllimport)
# endif
#else
# define HELPER_EXPORT
#endif
