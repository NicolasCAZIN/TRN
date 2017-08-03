#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(INITIALIZER_LIB)
#  define INITIALIZER_EXPORT Q_DECL_EXPORT
# else
#  define INITIALIZER_EXPORT Q_DECL_IMPORT
# endif
#else
# define INITIALIZER_EXPORT
#endif
