#pragma once

#include "network_global.h"

#include "Connection.h"

namespace TRN
{
	namespace Network
	{
		class NETWORK_EXPORT Manager : public 		boost::asio::io_service
		{
		private :
			class Handle;
			mutable std::unique_ptr<Handle> handle;

		public :
			Manager();
			~Manager();

		public :
			void stop();
			void start(const std::shared_ptr<TRN::Network::Connection> connection);
			void stop(const std::shared_ptr<TRN::Network::Connection> connection);
		public :
			static std::shared_ptr<Manager> create();
		};
	};
};
