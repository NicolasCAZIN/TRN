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
			Acceptor(const std::string &address, const std::string &port,
				const std::function <const std::shared_ptr<TRN::Network::Connection>(boost::asio::ip::tcp::socket, const std::shared_ptr<TRN::Network::Manager>)> &create
				);

			~Acceptor();

		public :
			void run();

		private :
			void accept();
			void await_stop();

		public :
			static std::shared_ptr<Acceptor> create(const std::string &address, const std::string &port,
				const std::function <const std::shared_ptr<TRN::Network::Connection>(boost::asio::ip::tcp::socket, const std::shared_ptr<TRN::Network::Manager>)> &create
				);
		};
	};
};

