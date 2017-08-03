#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(GPU_LIB)
#  define GPU_EXPORT Q_DECL_EXPORT
# else
#  define GPU_EXPORT Q_DECL_IMPORT
# endif
#else
# define GPU_EXPORT
#endif
