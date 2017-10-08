#pragma once



#ifndef TRN_STATIC
# if defined(LOCAL_LIB)
#  define LOCAL_EXPORT __declspec(dllexport)
# else
#  define LOCAL_EXPORT __declspec(dllimport)
# endif
#else
# define LOCAL_EXPORT
#endif
