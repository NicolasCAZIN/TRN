#pragma once

#include "Manager.h"
#include "Connection.h"

namespace TRN
{
	namespace Network
	{
		class NETWORK_EXPORT Acceptor
		{
		private :
			class Handle;
			mutable std::unique_ptr<Handle> handle;

		public :
			Acceptor(const std::string &address, const  unsigned short &port,
				const std::function <void(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection)> &on_accept
				);

			~Acceptor();

		public :
			void run();

		private :
			void accept();
			void await_stop();

		public :
			static std::shared_ptr<Acceptor> create(const std::string &address, const  unsigned short &port,
				const std::function <void(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection)> &on_accept
				);
		};
	};
};

