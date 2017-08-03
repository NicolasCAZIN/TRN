#pragma once

#include <QtCore/qglobal.h>

#ifndef BUILD_STATIC
# if defined(SIMULATOR_LIB)
#  define SIMULATOR_EXPORT Q_DECL_EXPORT
# else
#  define SIMULATOR_EXPORT Q_DECL_IMPORT
# endif
#else
# define SIMULATOR_EXPORT
#endif
