#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(RESERVOIR_LIB)
#  define RESERVOIR_EXPORT Q_DECL_EXPORT
# else
#  define RESERVOIR_EXPORT Q_DECL_IMPORT
# endif
#else
# define RESERVOIR_EXPORT
#endif
