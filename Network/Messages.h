#pragma once

#include "network_global.h"

namespace TRN
{
	namespace Network
	{
		struct  Data
		{
			std::string payload;
			int tag;
			int destination;


			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & payload;
				ar & tag;
				ar & destination;
		
			}
		};

		struct  Device
		{
			unsigned int index;
			std::string name;
			float glops;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & index;
				ar & name;
				ar & glops;
			}
		};

		struct  Processor
		{
			std::string name;
			std::vector<Device> devices;

			template<class Archive>
			void serialize(Archive & ar, const unsigned int version)
			{
				ar & name;
				ar & devices;
			}
		};
	};
};
