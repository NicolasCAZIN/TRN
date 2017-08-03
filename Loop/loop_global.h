#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(LOOP_LIB)
#  define LOOP_EXPORT Q_DECL_EXPORT
# else
#  define LOOP_EXPORT Q_DECL_IMPORT
# endif
#else
# define LOOP_EXPORT
#endif
