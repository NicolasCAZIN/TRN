#pragma once



#ifndef TRN_STATIC
# if defined(LOOP_LIB)
#  define LOOP_EXPORT __declspec(dllexport)
# else
#  define LOOP_EXPORT __declspec(dllimport)
# endif
#else
# define LOOP_EXPORT
#endif
