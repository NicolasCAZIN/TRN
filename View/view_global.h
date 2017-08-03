#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(VIEW_LIB)
#  define VIEW_EXPORT Q_DECL_EXPORT
# else
#  define VIEW_EXPORT Q_DECL_IMPORT
# endif
#else
# define VIEW_EXPORT
#endif
