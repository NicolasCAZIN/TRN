#pragma once

#ifndef BUILD_STATIC
# if defined(DISPLAY_LIB)
#  define DISPLAY_EXPORT __declspec(dllexport)
# else
#  define DISPLAY_EXPORT __declspec(dllimport)
# endif
#else
# define DISPLAY_EXPORT
#endif
