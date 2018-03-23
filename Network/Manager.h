#pragma once

#include "network_global.h"

#include "Connection.h"


namespace TRN
{
	namespace Network
	{
		class NETWORK_EXPORT Manager 
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Manager();
			virtual ~Manager();

		public :
			boost::asio::io_service &get_io_service();

		public :
			
			void stop();
			void start(const std::shared_ptr<TRN::Network::Connection> connection);
			void stop(const std::shared_ptr<TRN::Network::Connection> connection);
		public :
			static std::shared_ptr<Manager> create();
		};
	};
};
