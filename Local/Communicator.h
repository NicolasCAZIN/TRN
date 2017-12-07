#pragma once

#include "local_global.h"

#include "Engine/Communicator.h"
#include "Engine/Worker.h"

namespace TRN
{
	namespace Local
	{
		class LOCAL_EXPORT Communicator : public TRN::Engine::Communicator
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		public :
			Communicator(const int &max_rank);
			virtual ~Communicator();

		public :
			void append(std::shared_ptr<TRN::Engine::Worker> &worker);
			virtual void dispose() override;
		protected :
			virtual int rank() override;
			virtual std::size_t size() override;
			virtual void send(const int &destination, const TRN::Engine::Tag &tag, const std::string &data) override;
			virtual std::string receive(const int &destination, const TRN::Engine::Tag &tag) override;
			virtual boost::optional<TRN::Engine::Tag> probe(const int &destination) override;

		public :
			static std::shared_ptr<TRN::Local::Communicator> create(const int &max_rank);
		};
	};
};