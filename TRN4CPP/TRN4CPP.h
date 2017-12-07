#pragma once

#include "trn4cpp_global.h"

#ifdef TRN4CPP_EXTENDED
#include "Extended.h"
#else
#include "Simplified.h"
#include "Advanced.h"
#ifdef TRN4CPP_CALLBACKS
#include "Callbacks.h"
#endif
#ifdef TRN4CPP_CUSTOM
#include "Custom.h"
#endif
#endif

