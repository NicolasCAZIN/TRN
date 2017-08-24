#pragma once
#include "engine_global.h"
namespace TRN
{
	namespace Engine
	{
		enum Compressor
		{
			RAW,
			ZLIB,
			GZIP,
			BZIP2
		};

		template <const enum Compressor compressor>
		std::string ENGINE_EXPORT compress(const std::string &data);
		template <const enum Compressor compressor>
		std::string ENGINE_EXPORT decompress(const std::string &data);
	};
};
