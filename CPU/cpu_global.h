#pragma once



#ifndef TRN_STATIC
# if defined(CPU_LIB)
#  define CPU_EXPORT __declspec(dllexport)
# else
#  define CPU_EXPORT __declspec(dllimport)
# endif
#else
# define CPU_EXPORT
#endif
