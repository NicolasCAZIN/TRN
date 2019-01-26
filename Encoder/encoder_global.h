#pragma once

#ifndef BUILD_STATIC
# if defined(ENCODER_LIB)
#  define ENCODER_EXPORT __declspec(dllexport)
# else
#  define ENCODER_EXPORT __declspec(dllimport)
# endif
#else
# define ENCODER_EXPORT
#endif
