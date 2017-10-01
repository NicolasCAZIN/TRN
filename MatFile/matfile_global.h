#pragma once

#include <QtCore/qglobal.h>

#ifndef BUILD_STATIC
# if defined(MATFILE_LIB)
#  define MATFILE_EXPORT Q_DECL_EXPORT
# else
#  define MATFILE_EXPORT Q_DECL_IMPORT
# endif
#else
# define MATFILE_EXPORT
#endif
