#pragma once

#include "remote_global.h"
#include "Network/Manager.h"
#include "Network/Connection.h"
#include "Engine/Communicator.h"
#include "Engine/Worker.h"

namespace TRN
{
	namespace Remote
	{
		class REMOTE_EXPORT Communicator : public TRN::Engine::Communicator
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Communicator(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection, const int &rank, const std::size_t &size);
			virtual ~Communicator();

		public :
			virtual void start() override;
			virtual void stop() override;
			virtual void synchronize() override;

		public:
			virtual int rank() override;
			virtual std::size_t size() override;
			virtual void send(const int &destination, const TRN::Engine::Tag &tag, const std::string &data) override;
			virtual std::string receive(const int &destination, const TRN::Engine::Tag &tag) override;
			virtual boost::optional<TRN::Engine::Tag> probe(const int &destination) override;
		
		public:
			static std::shared_ptr<TRN::Remote::Communicator> create(const std::shared_ptr<TRN::Network::Manager> &manager, const std::shared_ptr<TRN::Network::Connection> &connection, const int &rank, const std::size_t &size);
		};
	};
};