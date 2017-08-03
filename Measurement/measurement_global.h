#pragma once

#include <QtCore/qglobal.h>

#ifndef BUILD_STATIC
# if defined(MEASUREMENT_LIB)
#  define MEASUREMENT_EXPORT Q_DECL_EXPORT
# else
#  define MEASUREMENT_EXPORT Q_DECL_IMPORT
# endif
#else
# define MEASUREMENT_EXPORT
#endif
