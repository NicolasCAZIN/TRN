#include "stdafx.h"
#include "Cache_impl.h"
#include "Helper/Logger.h"

/*#define SEGMENT_MUTEX "TRN_cache_mutex"*/
#define SEGMENT_IDENTIFIER "TRN_cache_segment"
#define CHECKSUMS_IDENTIFIER "TRN_cache_map"
#define COUNTER_IDENTIFIER "TRN_cache_counter"

#define GB(x) ((x) << 30)
#define MB(x) ((x) << 20)
#define KB(x) ((x) << 10)

typedef boost::interprocess::allocator<float, boost::interprocess::managed_shared_memory::segment_manager> ShmVectorAllocator;
typedef boost::interprocess::vector<float, ShmVectorAllocator> Vector;
typedef boost::interprocess::allocator<unsigned int, boost::interprocess::managed_shared_memory::segment_manager> ShmSetAllocator;
typedef boost::interprocess::set<unsigned int, std::less<unsigned int>, ShmSetAllocator> Set;


typedef unsigned int    KeyType;
typedef boost::interprocess::offset_ptr<Vector> MappedType;
typedef std::pair<const KeyType, MappedType> EntryType;
//allocator of for the map.
typedef boost::interprocess::allocator<EntryType, boost::interprocess::managed_shared_memory::segment_manager> ShmEntryAllocator;
typedef boost::interprocess::map<KeyType, MappedType, std::less<KeyType>, ShmEntryAllocator> Map;

 static boost::interprocess::managed_windows_shared_memory segment(boost::interprocess::open_or_create, SEGMENT_IDENTIFIER, GB(1) + MB(500));
static unsigned int process_id()
{
#ifdef _WIN32
	return GetCurrentProcessId();
#else
	return ::getpid();
#endif
}

static const std::string compute_key(const unsigned int checksum)
{
	return "checksum_" + std::to_string(checksum);
}
void TRN::Engine::Cache::initialize()
{
	/*WARNING_LOGGER << "initializing cache";
	if (boost::interprocess::shared_memory_object::remove(SEGMENT_IDENTIFIER))
		WARNING_LOGGER << SEGMENT_IDENTIFIER << " shared memory object was removed";
	else
		WARNING_LOGGER << SEGMENT_IDENTIFIER << " shared memory object was not removed";*/

	//auto pid = process_id();
/*	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> guard(*segment.find_or_construct<boost::interprocess::interprocess_mutex>(SEGMENT_MUTEX)());*/
	/*auto counter = segment.find_or_construct<Set>(COUNTER_IDENTIFIER)(segment.get_segment_manager());
	auto size = counter->size();
	
	counter->insert(pid);
	DEBUG_LOGGER << "processes attached to shared memory cache : " << counter->size();*/
/*	segment.atomic_func([&]() 
	{

	});*/
	//boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> guard(*segment.find_or_construct<boost::interprocess::interprocess_mutex>(SEGMENT_MUTEX)());


//	auto set = segment.find_or_construct<Set>(SET_IDENTIFIER)();
	


}
void TRN::Engine::Cache::uninitialize()
{



	/*WARNING_LOGGER << "uninitializing cache";
	if (boost::interprocess::shared_memory_object::remove(SEGMENT_IDENTIFIER))
		WARNING_LOGGER << SEGMENT_IDENTIFIER << " shared memory object was removed";
	else
		WARNING_LOGGER << SEGMENT_IDENTIFIER << " shared memory object was not removed";*/
	/*boost::interprocess::managed_shared_memory segment(boost::interprocess::open_or_create, SEGMENT_IDENTIFIER, GB(1));
	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> guard(*segment.find_or_construct<boost::interprocess::interprocess_mutex>(SEGMENT_MUTEX)());
	auto &counter = *segment.find_or_construct<Set>(COUNTER_IDENTIFIER)(segment.get_segment_manager());
	counter.erase(process_id());
	DEBUG_LOGGER << "processes attached to shared memory cache : " << counter.size();

	if (counter.empty())
	{
		DEBUG_LOGGER << "No more processes attached to shared memory cache. Destroying it";

	}
	else
	{
		DEBUG_LOGGER << counter.size() << " processes are still attached to shared memory cache. Won't do anything";
	}*/
}

TRN::Engine::Cache::Cache() : handle(std::make_unique<Handle>())
{

	
}

TRN::Engine::Cache::~Cache()
{
}

std::set<unsigned int> TRN::Engine::Cache::cached()
{
	std::set<unsigned int> result;
	segment.atomic_func([&]()
	{
		auto map = segment.find_or_construct<Map>(CHECKSUMS_IDENTIFIER)(segment.get_segment_manager());
		for (auto entry : *map)
		{
			result.insert(entry.first);
		}

	});
	return result;
}

void TRN::Engine::Cache::store(const unsigned int &checksum, const std::vector<float> &sequence)
{
	
	segment.atomic_func([&]()
	{
	//	boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex> guard(*segment.find_or_construct<boost::interprocess::interprocess_mutex>(SEGMENT_MUTEX)());
		auto map = segment.find_or_construct<Map>(CHECKSUMS_IDENTIFIER)(segment.get_segment_manager());

		if (map->find(checksum) != map->end())
		{
			DEBUG_LOGGER << "Data is already cached. Nothing will be updated";
		}
		else
		{
			auto free_memory = segment.get_free_memory();
			DEBUG_LOGGER << "Data having checksum 0x" << std::hex << checksum << " is not cached. Storing in shared memory cache";
			auto cached = segment.construct<Vector>(boost::interprocess::anonymous_instance)(sequence.begin(), sequence.end(), segment.get_segment_manager());
			map->emplace(checksum, cached);
			DEBUG_LOGGER << "Free shared memory " << free_memory;
		}
	});

}

bool	TRN::Engine::Cache::contains(const unsigned int &checksum)
{
	bool result = false;

	segment.atomic_func([&]()
	{
		auto map = segment.find_or_construct<Map>(CHECKSUMS_IDENTIFIER)(segment.get_segment_manager());
		result = map->find(checksum) != map->end();
	});
	return result;
}

std::vector<float> TRN::Engine::Cache::retrieve(const unsigned int &checksum)
{
	std::vector<float> result;
	
	segment.atomic_func([&]()
	{
		auto map = segment.find_or_construct<Map>(CHECKSUMS_IDENTIFIER)(segment.get_segment_manager());
		if (map->find(checksum) == map->end())
			throw std::runtime_error("Data is not cached");
		
		auto cached = map->at(checksum);
		result.resize(cached->size());
		std::copy(cached->begin(), cached->end(), result.begin());
	});
	

	return result;
}

std::shared_ptr<TRN::Engine::Cache> TRN::Engine::Cache::create()
{
	return std::make_shared<TRN::Engine::Cache>();
}

