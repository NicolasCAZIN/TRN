#pragma once

#include <QtCore/qglobal.h>

#ifndef BUILD_STATIC
# if defined(NETWORK_LIB)
#  define NETWORK_EXPORT Q_DECL_EXPORT
# else
#  define NETWORK_EXPORT Q_DECL_IMPORT
# endif
#else
# define NETWORK_EXPORT
#endif

