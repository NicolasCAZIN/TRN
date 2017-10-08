#pragma once

#ifndef TRN_STATIC
# if defined(CORE_LIB)
#  define CORE_EXPORT __declspec(dllexport)
# else
#  define CORE_EXPORT __declspec(dllimport)
# endif
#else
# define CORE_EXPORT
#endif
