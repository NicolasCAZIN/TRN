#pragma once



#ifndef BUILD_STATIC
# if defined(TRN4CPP_LIB)
#  define TRN4CPP_EXPORT __declspec(dllexport)
# else
#  define TRN4CPP_EXPORT __declspec(dllimport)
# endif
#else
# define TRN4CPP_EXPORT
#endif
