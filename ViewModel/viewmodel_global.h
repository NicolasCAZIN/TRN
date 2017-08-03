#pragma once

#include <QtCore/qglobal.h>

#ifndef QT_STATIC
# if defined(VIEWMODEL_LIB)
#  define VIEWMODEL_EXPORT Q_DECL_EXPORT
# else
#  define VIEWMODEL_EXPORT Q_DECL_IMPORT
# endif
#else
# define VIEWMODEL_EXPORT
#endif
