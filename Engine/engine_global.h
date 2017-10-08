#pragma once



#ifndef TRN_STATIC
# if defined(ENGINE_LIB)
#  define ENGINE_EXPORT __declspec(dllexport)
# else
#  define ENGINE_EXPORT __declspec(dllimport)
# endif
#else
# define ENGINE_EXPORT
#endif
