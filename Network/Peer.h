#pragma once

#include "network_global.h"

namespace TRN
{
	namespace Network
	{
		class NETWORK_EXPORT Peer
		{
		public :
			virtual void start() = 0;
			virtual void stop() = 0;
		};
	};
};