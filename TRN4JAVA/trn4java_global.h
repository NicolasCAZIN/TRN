#pragma once

#include <QtCore/qglobal.h>

#ifndef BUILD_STATIC
# if defined(TRN4JAVA_LIB)
#  define TRN4JAVA_EXPORT Q_DECL_EXPORT
# else
#  define TRN4JAVA_EXPORT Q_DECL_IMPORT
# endif
#else
# define TRN4JAVA_EXPORT
#endif
