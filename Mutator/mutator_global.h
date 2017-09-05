#pragma once

#include <QtCore/qglobal.h>

#ifndef BUILD_STATIC
# if defined(MUTATOR_LIB)
#  define MUTATOR_EXPORT Q_DECL_EXPORT
# else
#  define MUTATOR_EXPORT Q_DECL_IMPORT
# endif
#else
# define MUTATOR_EXPORT
#endif
