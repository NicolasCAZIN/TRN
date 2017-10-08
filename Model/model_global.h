#pragma once



#ifndef TRN_STATIC
# if defined(MODEL_LIB)
#  define MODEL_EXPORT __declspec(dllexport)
# else
#  define MODEL_EXPORT __declspec(dllimport)
# endif
#else
# define MODEL_EXPORT
#endif
