#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(REMOTE_LIB)
#  define REMOTE_EXPORT Q_DECL_EXPORT
# else
#  define REMOTE_EXPORT Q_DECL_IMPORT
# endif
#else
# define REMOTE_EXPORT
#endif
