#pragma once

#ifndef BUILD_STATIC
# if defined(DECODER_LIB)
#  define DECODER_EXPORT __declspec(dllexport)
# else
#  define DECODER_EXPORT __declspec(dllimport)
# endif
#else
# define CORE_EXPORT
#endif

