#include "stdafx.h"
#include "Connection_impl.h"
#include "Manager.h"
#include "Helper/Logger.h"

namespace atlas {

	typedef std::true_type single_parameter_pack_tag;
	typedef std::false_type not_single_parameter_pack_tag;
	typedef std::true_type last_parameter_tag;
	typedef std::false_type not_last_parameter_tag;

	namespace {

		template<typename ... Elements>
		struct __is_single_parameter_pack_helper {
			typedef typename std::conditional<1 == sizeof...(Elements), std::true_type, std::false_type>::type type;
		};

		template<std::size_t idx, typename ... Elements>
		struct __is_last_parameter_helper {
			typedef typename std::conditional<idx + 1 == sizeof...(Elements)-1, std::true_type, std::false_type>::type type;
		};

	}

	template<typename ... Elements>
	struct is_single_parameter_pack :
		public std::integral_constant<bool, __is_single_parameter_pack_helper<Elements...>::type::value>
	{};

	template<std::size_t idx, typename ... Elements>
	struct is_last_parameter :
		public std::integral_constant<bool, __is_last_parameter_helper<idx, Elements...>::type::value>
	{};

	//  template<typename F, typename ...Args>
	//  struct is_void_call : public std::is_void<std::result_of<F(Args...)>::type>::type {
	//  };

} // atlas

namespace atlas {
	namespace {

		template<std::size_t idx, typename Archive, typename ... Elements>
		void aux_serialize(Archive& ar, std::tuple<Elements...>& t, single_parameter_pack_tag) {
			ar & std::get<idx>(t);
		}

		template<std::size_t idx, typename Archive, typename ... Elements>
		void aux_serialize(Archive& ar, std::tuple<Elements...>& t, not_single_parameter_pack_tag) {
			ar & std::get<idx>(t);

			aux_serialize<idx + 1>(ar, t, atlas::is_last_parameter<idx, Elements...>());
		}

		template<typename Archive, typename ... Elements>
		void serialize(Archive& ar, std::tuple<Elements...>& t, last_parameter_tag) {
			ar & std::get<0>(t);
		}

		template<typename Archive, typename ... Elements>
		void serialize(Archive& ar, std::tuple<Elements...>& t, not_last_parameter_tag) {
			aux_serialize<0>(ar, t, std::false_type());
		}
	}

} // atlas

namespace boost {
	namespace serialization {

		template<typename Archive, typename ... Elements>
		Archive& serialize(Archive& ar, std::tuple<Elements...>& t, const int version) {
			atlas::serialize(ar, t, atlas::is_single_parameter_pack<Elements...>());

			return ar;
		}

	} // serialization
} // boost


TRN::Network::Connection::Connection(const std::shared_ptr<TRN::Network::Manager> manager, boost::asio::ip::tcp::socket socket) :
	handle(std::make_unique<Handle>(manager, std::move(socket)))
{
}


TRN::Network::Connection::Connection(const std::shared_ptr<TRN::Network::Manager> manager) :
	handle(std::make_unique<Handle>(manager))
{
}

TRN::Network::Connection::~Connection()


{
	handle.reset();
}


void TRN::Network::Connection::initialize()
{
	std::cout << "Starting connection by invoking manager" << std::endl;

	handle->manager->start(std::shared_ptr<TRN::Network::Connection>(this, [=](void *ptr) {}));//hared_from_this());
}

void TRN::Network::Connection::uninitialize()
{
	handle->manager->stop(std::shared_ptr<TRN::Network::Connection>(this, [=](void *ptr) {}));//hared_from_this());
}

void TRN::Network::Connection::start()
{
	std::cout << "Waiting for command" << std::endl;
	read_command();
}
void TRN::Network::Connection::stop()
{
	std::cout << "Closing connection" << std::endl;
	handle->socket.close();
}
#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
template <typename Data>
void TRN::Network::Connection::send(const std::string &type, const Data &data)
{
	boost::asio::streambuf b;
	
	std::ostream  archive_stream(&b);
	boost::archive::binary_oarchive archive(archive_stream);
	
	archive << data;

	
	auto size = std::to_string(b.size());
	int id = std::get<0>(data);
	
	send(id, { "DATA", size, type });

	boost::asio::write(handle->socket, b);
	std::cout << size << " bytes sent" << std::endl;
}

void TRN::Network::Connection::send(const unsigned int &id, const std::vector<std::string> &command)
{
	std::string str = std::to_string(id) + " " + boost::algorithm::join(command, " ") + "\r\n";

	std::cout << "SENDING : " << str << std::endl;
	boost::asio::write(handle->socket, boost::asio::buffer(str));

		//boost::bind(&TRN::Network::Connection::write_command_handle, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
}

void TRN::Network::Connection::send(const unsigned int &id, const std::string &label, const std::vector<float> &elements, const std::size_t &rows, const std::size_t &cols)
{
	send("MATRIX", std::make_tuple(id, label, elements, rows, cols));
}

void TRN::Network::Connection::send(const unsigned int &id, const std::vector<int> &offsets, const std::vector<int> &durations)
{
	send("SCHEDULING", std::make_tuple(id, offsets, durations));
}

void TRN::Network::Connection::read_command()
{
	std::cout << "launching async read command" << handle->rx.size() << std::endl;

	boost::asio::async_read_until(handle->socket, handle->rx, "\r\n",
		boost::bind(&TRN::Network::Connection::read_command_handle, this, boost::asio::placeholders::error, boost::asio::placeholders::bytes_transferred));
}
template <typename Data>
void TRN::Network::Connection::read_data(const std::size_t &expected, const std::function<void(const Data &data)> &callback)
{
	std::cout << "expecting " << expected << " bytes" << std::endl;

	std:: cout << "pending " << handle->rx.size() << std::endl;

	handle->expected = expected;
	std::size_t remaining = 0;
	if (expected > handle->rx.size())
		remaining = expected - handle->rx.size();

	std::cout << "remaining " << remaining << " bytes" << std::endl;
	boost::asio::async_read(handle->socket, handle->rx,
		boost::asio::transfer_exactly(remaining),
		[this, callback](const boost::system::error_code &error, const std::size_t &bytes_transferred)
	{
		TRN::Network::Connection::read_data_handle<Data>(error, bytes_transferred, callback);
	});
}

void TRN::Network::Connection::read_command_handle(const boost::system::error_code &error, const std::size_t &bytes_transferred)
{
	try
	{
		std::cout << "PEER received : " << bytes_transferred << std::endl;
		if (error)
		{
			throw std::runtime_error(error.message());
		}

		std::size_t remaining = handle->rx.in_avail();

		std::istream is(&handle->rx);
        std::string line;
		std::getline(is, line);

		if (!line.empty() && line[line.size() - 1] == '\r')
			line.erase(line.size() - 1);
		boost::trim(line);
		std::cout << "remaining " << remaining << std::endl;

		if (line.empty())
		{
			read_command();
		}
		else
		{
			std::cout << "PEER received : " << line << std::endl;
			//	LOG_INFO("Server received : " << line);
			std::vector<std::string> tokens;
			boost::split(tokens, line, boost::is_any_of("\t "));
			if (tokens.empty())
			{
				throw std::logic_error("received empty command : " + line);
			}


	
			//LOG_DEBUG("type is : " << type);
			std::cout << tokens[1] << std::endl;
			if (tokens[1] == "DATA")
			{
				//			LOG_INFO("received DATA");
				if (tokens.size() < 4)
				{
					throw std::logic_error("A least 2 tokens expected after DATA");
				}

				auto size = boost::lexical_cast<std::size_t>(tokens[2]);
				std::cout << "size to receive " << size << std::endl;
			
				auto subtype = boost::to_upper_copy(tokens[3]);
				std::cout << "type to receive " << subtype << std::endl;
				if (subtype == "MATRIX")
				{
					read_data<std::tuple<int, std::string, std::vector<float>, std::size_t, std::size_t>>(size, 
						[this](const std::tuple<int, std::string, std::vector<float>, std::size_t, std::size_t> &tuple)
					{
						receive_matrix(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple), std::get<3>(tuple), std::get<4>(tuple));
					});
				}
				else if (subtype == "SCHEDULING")
				{
					read_data<std::tuple<int, std::vector<int>, std::vector<int>>> (size,
						[this](const std::tuple<int, std::vector<int>, std::vector<int>> &tuple)
					{
						receive_scheduling(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
					});
				}
				else
				{
					throw std::logic_error("Unexpected subtype : " + subtype);
				}
			}
			else
			{
				auto id = boost::lexical_cast<int>(tokens[0]);

				receive_command(id, std::vector<std::string>(tokens.begin() + 1, tokens.end()));
				/*std::thread second([this, id, tokens]() 
				{
				});
				second.*/
				read_command();
			}
		}
		//std::cout << "data available " << handle->rx.in_avail() << std::endl;
	}
	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		uninitialize();

		//LOG_FATAL(e.what());
	}
}

template <typename Data>
void TRN::Network::Connection::read_data_handle(const boost::system::error_code &error, const std::size_t &bytes_transferred, const std::function<void(const Data &data)> &callback)
{
	try
	{
		if (error)
		{
			throw std::runtime_error(error.message());
		}
		std::cout << "transferred" << bytes_transferred << std::endl;
	/*	if (bytes_transferred < handle->expected)
		{
			throw std::logic_error("bytes transderred (" + std::to_string(bytes_transferred) + ") < expected bytes (" + std::to_string(handle->expected) + ")");
		}*/
		
		Data received;

		std::istream archive_stream(&handle->rx);
		boost::archive::binary_iarchive archive(archive_stream);

		archive >> received;
		//handle->rx.consume(bytes_transferred);
		handle->expected = 0;
		std::cout << "deserialized" << std::endl;
		callback(received);
		
		std::cout << "data available before reading command " << handle->rx.in_avail() << std::endl;
		read_command();
	}

	catch (std::exception &e)
	{
		std::cerr << e.what() << std::endl;
		uninitialize();
		//LOG_FATAL(e.what());
	}
}

void TRN::Network::Connection::write_command_handle(const boost::system::error_code &error, const std::size_t &bytes_transferred)
{
	try
	{
		if (error)
		{
			throw std::runtime_error(error.message());
		}

		std::cout << "command transmitted (" << bytes_transferred << ")" << std::endl;
		
	}
	catch (std::exception &e)
	{
		uninitialize();
		//LOG_FATAL(e.what());
	}
}
void TRN::Network::Connection::write_data_handle(const boost::system::error_code &error, const std::size_t &bytes_transferred)
{
	try
	{
		if (error)
		{
			throw std::runtime_error(error.message());
		}

		std::cout << "data transmitted (" << bytes_transferred << ")" << std::endl;
	}
	catch (std::exception &e)
	{

		uninitialize();
		//LOG_FATAL(e.what());
	}
}

