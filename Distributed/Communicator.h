#pragma once

#include "distributed_global.h"

#include "Engine/Communicator.h"
#include "Engine/Worker.h"

namespace TRN
{
	namespace Distributed
	{
		class DISTRIBUTED_EXPORT Communicator : public TRN::Engine::Communicator
		{
		private:
			class Handle;
			std::unique_ptr<Handle> handle;

		public:
			Communicator(int argc, char *argv[]);
			virtual ~Communicator();

		public :
			virtual void dispose() override;

		protected:
			virtual int rank() override;
			virtual std::size_t size() override;
			virtual void send(const int &destination, const TRN::Engine::Tag &tag, const std::string &data) override;
			virtual std::string receive(const int &destination, const TRN::Engine::Tag &tag) override;
			virtual boost::optional<TRN::Engine::Tag> probe(const int &destination) override;
	
		public:
			static std::shared_ptr<Communicator> create(int argc, char *argv[]);
		};
	};
};