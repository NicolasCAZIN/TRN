#pragma once

#ifdef USE_VLD
#include <vld.h>
#endif 

#include <SDKDDKVer.h>

#include <cstdlib>
#include <boost/tuple/tuple.hpp>
#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/multi_array.hpp>
#include <boost/dll/import.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/info_parser.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <strtk.hpp>
#include <vector>
#include <memory>
#include <functional>
#include <list>
#include <string>
#include <sstream>
#include <iostream>
#include <mutex>
#include <future>
#include <thread>
#include <queue>
#include <set>
#include <map>
#include <ctime>

#include <mat.h>