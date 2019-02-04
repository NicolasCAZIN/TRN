#include "stdafx.h"
#include "Cache_impl.h"
#include "Helper/Logger.h"

/*#define SEGMENT_MUTEX "TRN_cache_mutex"*/

#ifdef SHM_CACHE

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


static const std::string compute_key(const unsigned int checksum)
{
	return "checksum_" + std::to_string(checksum);
}

#else
		
static std::mutex mutex;
static std::map<unsigned int, std::vector<float>> cache;
#endif
void TRN::Engine::Cache::initialize()
{
}
void TRN::Engine::Cache::uninitialize()
{
}

TRN::Engine::Cache::Cache() : handle(std::make_unique<Handle>())
{
}

TRN::Engine::Cache::~Cache()
{
}

std::set<unsigned int> TRN::Engine::Cache::cached()
{
#ifdef SHM_CACHE
	segment.atomic_func([&]() -> std::set<unsigned int>
	{
		std::set<unsigned int> result;
		auto map = segment.find_or_construct<Map>(CHECKSUMS_IDENTIFIER)(segment.get_segment_manager());
		for (auto entry : *map)
		{
			result.insert(entry.first);
		}
		return result;
	});
#else
	std::unique_lock<std::mutex> lock(mutex);
	std::set<unsigned int> result;
	for (auto entry : cache)
	{
		result.insert(entry.first);
	}
	return result;
#endif
}

void TRN::Engine::Cache::store(const unsigned int &checksum, const std::vector<float> &sequence)
{
#ifdef SHM_CACHE
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
#else
	std::unique_lock<std::mutex> lock(mutex);

	cache[checksum] = sequence;
#endif
}

bool	TRN::Engine::Cache::contains(const unsigned int &checksum)
{
#if SHM_CACHE
	bool result = false;

	segment.atomic_func([&]()
	{
		auto map = segment.find_or_construct<Map>(CHECKSUMS_IDENTIFIER)(segment.get_segment_manager());
		result = map->find(checksum) != map->end();
	});
	return result;
#else
	std::unique_lock<std::mutex> lock(mutex);

	return cache.find(checksum) != cache.end();
#endif
}

std::vector<float> TRN::Engine::Cache::retrieve(const unsigned int &checksum)
{
#if SHM_CACHE
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
#else
	std::unique_lock<std::mutex> lock(mutex);

	return cache[checksum];
#endif
}

std::shared_ptr<TRN::Engine::Cache> TRN::Engine::Cache::create()
{
	return std::make_shared<TRN::Engine::Cache>();
}

