#pragma once

#ifdef USE_VLD
#include <vld.h>
#endif 
#include <boost/tuple/tuple.hpp>
#include <boost/asio.hpp>
#include <boost/function.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/utility.hpp>
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