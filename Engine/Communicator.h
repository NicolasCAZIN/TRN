#pragma once

#include "engine_global.h"
#include "Messages.h"
#include "Compressor.h"

namespace TRN
{
	namespace Engine
	{
		class ENGINE_EXPORT Communicator
		{
		private :
			class Handle;
			std::unique_ptr<Handle> handle;

		protected :
			Communicator();
		public :
			virtual ~Communicator();


		public :
			virtual void dispose() = 0;
			std::string host();
			virtual std::size_t size() = 0;
			virtual int rank() = 0;
			virtual boost::optional<TRN::Engine::Tag> probe(const int &destination) = 0;

			virtual void send(const int &destination, const TRN::Engine::Tag &tag, const std::string &data) = 0;
			virtual std::string receive(const int &destination, const TRN::Engine::Tag &tag) = 0;

			template <TRN::Engine::Tag tag>
			void send(const TRN::Engine::Message<tag> &message, const int &destination);
			
			template <TRN::Engine::Tag tag>
			void broadcast(const TRN::Engine::Message<tag> &message );

			template <TRN::Engine::Tag tag>
			TRN::Engine::Message<tag> receive(const int &destination);
		};
	};
};