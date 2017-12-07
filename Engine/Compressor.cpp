#include "stdafx.h"
#include "Compressor.h"
#include "Helper/Logger.h"

template<const enum TRN::Engine::Compressor compressor>
static inline std::string TRN::Engine::compress(const std::string &data)
{
	TRACE_LOGGER;
}
template<const enum Compressor compressor>
static inline std::string decompress(const std::string &data)
{
	TRACE_LOGGER;
}

template<>
std::string TRN::Engine::compress<TRN::Engine::Compressor::RAW>(const std::string &data)
{
	TRACE_LOGGER;
	return data;
}

template<>
std::string TRN::Engine::decompress<TRN::Engine::Compressor::RAW>(const std::string &data)
{
	TRACE_LOGGER;
	return data;
}

template<>
std::string TRN::Engine::compress<TRN::Engine::Compressor::BZIP2>(const std::string &data)
{
	TRACE_LOGGER;
	std::stringstream compressed;
	std::stringstream decompressed;
	decompressed << data;
	boost::iostreams::filtering_streambuf<boost::iostreams::input> out;
	out.push(boost::iostreams::bzip2_compressor());
	out.push(decompressed);
	boost::iostreams::copy(out, compressed);
	return compressed.str();
}

template<>
std::string TRN::Engine::decompress<TRN::Engine::Compressor::BZIP2>(const std::string &data)
{
	TRACE_LOGGER;
	std::stringstream compressed;
	std::stringstream decompressed;
	compressed << data;
	boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
	in.push(boost::iostreams::bzip2_decompressor());
	in.push(compressed);
	boost::iostreams::copy(in, decompressed);
	return decompressed.str();
}

template<>
std::string TRN::Engine::compress<TRN::Engine::Compressor::ZLIB>(const std::string &data)
{
	TRACE_LOGGER;
	std::stringstream compressed;
	std::stringstream decompressed;
	decompressed << data;
	boost::iostreams::filtering_streambuf<boost::iostreams::input> out;
	out.push(boost::iostreams::zlib_compressor());
	out.push(decompressed);
	boost::iostreams::copy(out, compressed);
	return compressed.str();
}


template<>
std::string TRN::Engine::decompress<TRN::Engine::Compressor::ZLIB>(const std::string &data)
{
	TRACE_LOGGER;
	std::stringstream compressed;
	std::stringstream decompressed;
	compressed << data;
	boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
	in.push(boost::iostreams::zlib_decompressor());
	in.push(compressed);
	boost::iostreams::copy(in, decompressed);
	return decompressed.str();
}

template<>
std::string TRN::Engine::compress<TRN::Engine::Compressor::GZIP>(const std::string &data)
{
	TRACE_LOGGER;
	std::stringstream compressed;
	std::stringstream decompressed;
	decompressed << data;
	boost::iostreams::filtering_streambuf<boost::iostreams::input> out;
	out.push(boost::iostreams::gzip_compressor());
	out.push(decompressed);
	boost::iostreams::copy(out, compressed);
	return compressed.str();
}

template<>
std::string TRN::Engine::decompress<TRN::Engine::Compressor::GZIP>(const std::string &data)
{
	TRACE_LOGGER;
	std::stringstream compressed;
	std::stringstream decompressed;
	compressed << data;
	boost::iostreams::filtering_streambuf<boost::iostreams::input> in;
	in.push(boost::iostreams::gzip_decompressor());
	in.push(compressed);
	boost::iostreams::copy(in, decompressed);
	return decompressed.str();
}
