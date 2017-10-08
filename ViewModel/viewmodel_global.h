#pragma once



#ifndef TRN_STATIC
# if defined(VIEWMODEL_LIB)
#  define VIEWMODEL_EXPORT __declspec(dllexport)
# else
#  define VIEWMODEL_EXPORT __declspec(dllimport)
# endif
#else
# define VIEWMODEL_EXPORT
#endif
