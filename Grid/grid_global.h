#pragma once

#ifndef BUILD_STATIC
# if defined(GRID_LIB)
#  define GRID_EXPORT __declspec(dllexport)
# else
#  define GRID_EXPORT __declspec(dllimport)
# endif
#else
# define GRID_EXPORT
#endif