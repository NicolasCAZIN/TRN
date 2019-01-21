#pragma once

#include <QtCore/qglobal.h>

#ifndef BUILD_STATIC
# if defined(ENCODER_LIB)
#  define ENCODER_EXPORT Q_DECL_EXPORT
# else
#  define ENCODER_EXPORT Q_DECL_IMPORT
# endif
#else
# define ENCODER_EXPORT
#endif
