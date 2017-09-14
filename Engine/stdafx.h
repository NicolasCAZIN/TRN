#pragma once
#ifdef USE_VLD
#include <vld.h>
#endif 

#include <boost/tuple/tuple.hpp>
#include <boost/asio.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/copy.hpp>

#include <vector>
#include <memory>
#include <functional>
#include <list>
#include <string>
#include <sstream>
#include <iostream>
#include <mutex>
#include <thread>
#include <queue>
#include <set>
#include <map>
#include <ctime>

class PrintThread : public std::ostringstream
{
public:
	PrintThread() = default;

	~PrintThread();

private:
	static std::mutex _mutexPrint;
};