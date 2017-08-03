#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(ENGINE_LIB)
#  define ENGINE_EXPORT Q_DECL_EXPORT
# else
#  define ENGINE_EXPORT Q_DECL_IMPORT
# endif
#else
# define ENGINE_EXPORT
#endif
