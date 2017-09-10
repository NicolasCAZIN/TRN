#pragma once

#include "network_global.h"
#include "Helper/Observer.h"
namespace TRN
{
	namespace Network
	{
		class Manager;
		class NETWORK_EXPORT Connection 
		{
		public :
			class Handle;
		protected:
			mutable std::unique_ptr<Handle> handle;

		public :
			Connection(const std::shared_ptr<TRN::Network::Manager> manager, boost::asio::ip::tcp::socket socket);
			Connection(const std::shared_ptr<TRN::Network::Manager> manager);
			virtual ~Connection();

		public :
			void start();
			void stop();
		protected :
			void initialize();
			void uninitialize();
		public :
			void send(const unsigned int &id, const std::vector<std::string> &command);
			void send(const unsigned int &id, const std::string &label, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols);
			void send(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations);

	

		private :
			template<typename Data>
			void send(const std::string &type, const Data &data);
			void read_command();

			template<typename Data>
			void read_data(const std::size_t &expected, const std::function<void(const Data &data)> &callback);

			void read_command_handle(const boost::system::error_code &error, const std::size_t &bytes_transferred);

			template<typename Data>
			void read_data_handle(const boost::system::error_code &error, const std::size_t &bytes_transferred, const std::function<void(const Data &data)> &callback);
			void write_command_handle(const boost::system::error_code &error, const std::size_t &bytes_transferred);
			void write_data_handle(const boost::system::error_code &error, const std::size_t &bytes_transferred);

		};
	};
};
