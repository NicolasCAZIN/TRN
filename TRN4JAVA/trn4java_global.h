#pragma once



#ifndef BUILD_STATIC
# if defined(TRN4JAVA_LIB)
#  define TRN4JAVA_EXPORT __declspec(dllexport)
# else
#  define TRN4JAVA_EXPORT __declspec(dllimport)
# endif
#else
# define TRN4JAVA_EXPORT
#endif
