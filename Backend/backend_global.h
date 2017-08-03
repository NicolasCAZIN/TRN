#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(BACKEND_LIB)
#  define BACKEND_EXPORT Q_DECL_EXPORT
# else
#  define BACKEND_EXPORT Q_DECL_IMPORT
# endif
#else
# define BACKEND_EXPORT
#endif
