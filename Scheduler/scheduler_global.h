#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(SCHEDULER_LIB)
#  define SCHEDULER_EXPORT Q_DECL_EXPORT
# else
#  define SCHEDULER_EXPORT Q_DECL_IMPORT
# endif
#else
# define SCHEDULER_EXPORT
#endif
