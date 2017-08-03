#pragma once

#include <QtCore/qglobal.h>

#ifndef BUILD_STATIC
# if defined(TRN4CPP_LIB)
#  define TRN4CPP_EXPORT Q_DECL_EXPORT
# else
#  define TRN4CPP_EXPORT Q_DECL_IMPORT
# endif
#else
# define TRN4CPP_EXPORT
#endif
