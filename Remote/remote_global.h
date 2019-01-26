#pragma once

#ifndef BUILD_STATIC
# if defined(REMOTE_LIB)
#  define REMOTE_EXPORT __declspec(dllexport)
# else
#  define REMOTE_EXPORT __declspec(dllimport)
# endif
#else
# define REMOTE_EXPORT
#endif
