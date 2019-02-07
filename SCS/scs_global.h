#pragma once

#ifndef BUILD_STATIC
# if defined(SCS_LIB)
#  define SCS_EXPORT __declspec(dllexport)
# else
#  define SCS_EXPORT __declspec(dllimport)
# endif
#else
# define MATFILE_EXPORT
#endif
