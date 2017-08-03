#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(LOCAL_LIB)
#  define LOCAL_EXPORT Q_DECL_EXPORT
# else
#  define LOCAL_EXPORT Q_DECL_IMPORT
# endif
#else
# define LOCAL_EXPORT
#endif
