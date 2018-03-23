#pragma once

#include <QtCore/qglobal.h>

#ifndef BUILD_STATIC
# if defined(DECODER_LIB)
#  define DECODER_EXPORT Q_DECL_EXPORT
# else
#  define DECODER_EXPORT Q_DECL_IMPORT
# endif
#else
# define DECODER_EXPORT
#endif
