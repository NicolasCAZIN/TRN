#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(HELPER_LIB)
#  define HELPER_EXPORT Q_DECL_EXPORT
# else
#  define HELPER_EXPORT Q_DECL_IMPORT
# endif
#else
# define HELPER_EXPORT
#endif
