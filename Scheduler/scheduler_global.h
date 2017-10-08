#pragma once



#ifndef TRN_STATIC
# if defined(SCHEDULER_LIB)
#  define SCHEDULER_EXPORT __declspec(dllexport)
# else
#  define SCHEDULER_EXPORT __declspec(dllimport)
# endif
#else
# define SCHEDULER_EXPORT
#endif
