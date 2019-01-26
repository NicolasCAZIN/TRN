#pragma once



#ifndef BUILD_STATIC
# if defined(PRESENTER_LIB)
#  define PRESENTER_EXPORT __declspec(dllexport)
# else
#  define PRESENTER_EXPORT __declspec(dllimport)
# endif
#else
# define VIEWMODEL_EXPORT
#endif
