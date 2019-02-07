#pragma once

#ifndef BUILD_STATIC
# if defined(CSV_LIB)
#  define CSV_EXPORT __declspec(dllexport)
# else
#  define CSV_EXPORT __declspec(dllimport)
# endif
#else
# define MATFILE_EXPORT
#endif
