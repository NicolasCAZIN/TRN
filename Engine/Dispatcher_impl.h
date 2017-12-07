#pragma once

#include "Dispatcher.h"

class TRN::Engine::Dispatcher::Handle 
{
public :
	struct Processor
	{
		std::string name;
		std::string host;
		unsigned int index;
		int rank;
		Processor()
		{

		}
		Processor(int rank, unsigned int index, std::string host, std::string name) :
			rank(rank),
			index(index),
			host(host),
			name(name)
		{

		}
		Processor  &operator = (const Processor &processor)
		{
			rank = processor.rank;
			index = processor.index;
			host = processor.host;
			name = processor.name;

			return *this;
		}
		bool operator == (const Processor &processor)
		{
			return rank == processor.rank && index == processor.index && host == processor.host && name == processor.name;
		}
		bool operator != (const Processor &processor)
		{
			return !operator ==(processor);
		}
		static friend std::ostream &operator << (std::ostream &stream, TRN::Engine::Dispatcher::Handle::Processor &processor)
		{
			stream << "'" << processor.name << "'" << " #" << processor.index << " @ " << processor.host << " / " << processor.rank;
			return stream;
		}
	};

public :
	std::map<unsigned short, std::shared_ptr<TRN::Engine::Communicator>> to_frontend;


	std::map<int, Processor> processors;
	std::mutex mutex;
};