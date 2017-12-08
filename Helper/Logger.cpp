#include "stdafx.h"
#include "Logger.h"
#include <windows.h>

static std::mutex mutex{};

#define BLACK			0
#define BLUE			1
#define GREEN			2
#define CYAN			3
#define RED				4
#define MAGENTA			5
#define BROWN			6
#define LIGHTGRAY		7
#define DARKGRAY		8
#define LIGHTBLUE		9
#define LIGHTGREEN		10
#define LIGHTCYAN		11
#define LIGHTRED		12
#define LIGHTMAGENTA	13
#define YELLOW			14
#define WHITE			15

#define TEXT_COLOR(ForgC, BackC) ((WORD)


struct Color
{
	Color( int foreground, int background, bool intensity = false) : value
	(
			(((background) & 0x0F) << 4) | 
			((foreground) & 0x0F) | 
			(intensity ? (FOREGROUND_INTENSITY) : 0)
	)
	{

	}
	WORD value;
};
static const Color NEUTRAL_COLOR(WHITE, BLACK);

static const Color DATE_COLOR(YELLOW, BLUE);
static const Color MODULE_COLOR(CYAN, BLUE);
static const Color MESSAGE_COLOR(WHITE, BLACK, true);

static const Color TRACE_COLOR(LIGHTGRAY, CYAN, true);
static const Color DEBUG_COLOR(LIGHTGRAY, DARKGRAY, true);
static const Color INFORMATION_COLOR(WHITE, GREEN, true);
static const Color WARNING_COLOR(RED, YELLOW, true);
static const Color ERROR_COLOR(WHITE, RED, true);

static inline std::ostream& operator<<(std::ostream& os, const Color &color)
{
	if (os.rdbuf() == std::cerr.rdbuf())
		SetConsoleTextAttribute(GetStdHandle(STD_ERROR_HANDLE), color.value);
	else
	
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color.value);
	return os;
}

static inline std::ostream& operator<<(std::ostream& os, const TRN::Helper::Logger::Severity &severity)
{
	switch (severity)
	{
		case TRN::Helper::Logger::Severity::TRACE_LEVEL:
			os << TRACE_COLOR << "TRACE      ";
			break;
		case TRN::Helper::Logger::Severity::DEBUG_LEVEL:
			os << DEBUG_COLOR <<  "DEBUG      ";
			break;
		case TRN::Helper::Logger::Severity::INFORMATION_LEVEL:
			os << INFORMATION_COLOR << "INFORMATION";
			break;
		case TRN::Helper::Logger::Severity::WARNING_LEVEL:
			os << WARNING_COLOR << "WARNING    ";
			break;
		case TRN::Helper::Logger::Severity::ERROR_LEVEL:
			os << ERROR_COLOR << "ERROR      ";
			break;
	}

	return os;
}

static std::ostream &stream(const TRN::Helper::Logger::Severity &severity)
{
	if (severity >= TRN::Helper::Logger::Severity::WARNING_LEVEL)
		return std::cerr;
	else
		return std::cout;
}
// helper class

static std::string hostname;
static TRN::Helper::Logger::Severity level = TRN::Helper::Logger::Severity::DEBUG_LEVEL;

TRN::Helper::Logger::Logger(const TRN::Helper::Logger::Severity &severity, const std::string &module) : severity(severity), module(module)
{
	if (hostname.empty())
		hostname = boost::asio::ip::host_name();
}

TRN::Helper::Logger::~Logger()
{
	std::lock_guard<std::mutex> guard(mutex);
	if (severity >= level)
	{
		auto now = boost::posix_time::microsec_clock::universal_time();
		stream(severity) << severity << NEUTRAL_COLOR << " " << DATE_COLOR << "[" << now << "]" << NEUTRAL_COLOR << " " << MODULE_COLOR << module << "@" << hostname << NEUTRAL_COLOR << " -> " << MESSAGE_COLOR << this->str() << std::endl << std::flush;
	}
}



void TRN::Helper::Logger::setup(const TRN::Helper::Logger::Severity &severity)
{
	std::lock_guard<std::mutex> guard(mutex);
	level = severity;

}

std::istream &std::operator >> (std::istream &is, TRN::Helper::Logger::Severity &severity)
{
	std::string token;

	is >> token;
	boost::to_upper(token);
	if (token == "ERROR")
	{
		severity = TRN::Helper::Logger::Severity::ERROR_LEVEL;
	}
	else if (token == "WARNING")
	{
		severity = TRN::Helper::Logger::Severity::WARNING_LEVEL;
	}
	else if (token == "INFORMATION")
	{
		severity = TRN::Helper::Logger::Severity::INFORMATION_LEVEL;
	}
	else if (token == "DEBUG")
	{
		severity = TRN::Helper::Logger::Severity::DEBUG_LEVEL;
	}
	else if (token == "TRACE")
	{
		severity = TRN::Helper::Logger::Severity::TRACE_LEVEL;
	}
	else
	{
		throw std::invalid_argument("Unexpected token " + token);
	}
	return is;
}

/*BOOST_LOG_GLOBAL_LOGGER_CTOR_ARGS(sysLogger,
	boost::log::sources::severity_channel_logger_mt<boost::log::trivial::severity_level>,
	(boost::log::keywords::channel = "SYSLF"));

BOOST_LOG_GLOBAL_LOGGER_CTOR_ARGS(dataLogger,
	boost::log::sources::severity_channel_logger_mt<boost::log::trivial::severity_level>,
	(boost::log::keywords::channel = "DATALF"));

// Custom formatter factory to add TimeStamp format support in config ini file.
// Allows %TimeStamp(format=\"%Y.%m.%d %H:%M:%S.%f\")% to be used in ini config file for property Format.
class TimeStampFormatterFactory :
	public boost::log::basic_formatter_factory<char, boost::posix_time::ptime>
{
public:
	formatter_type create_formatter(const boost::log::attribute_name& name, const args_map& args) {
		args_map::const_iterator it = args.find("format");
		if (it != args.end()) {
			return boost::log::expressions::stream
				<< boost::log::expressions::format_date_time<boost::posix_time::ptime>(
					boost::log::expressions::attr<boost::posix_time::ptime>(name), it->second);
		}
		else {
			return boost::log::expressions::stream
				<< boost::log::expressions::attr<boost::posix_time::ptime>(name);
		}
	}
};

// Custom formatter factory to add Uptime format support in config ini file.
// Allows %Uptime(format=\"%O:%M:%S.%f\")% to be used in ini config file for property Format.
// boost::log::attributes::timer value type is boost::posix_time::time_duration
class UptimeFormatterFactory :
	public boost::log::basic_formatter_factory<char, boost::posix_time::time_duration>
{
public:
	formatter_type create_formatter(const boost::log::attribute_name& name, const args_map& args)
	{
		args_map::const_iterator it = args.find("format");
		if (it != args.end()) {
			return boost::log::expressions::stream
				<< boost::log::expressions::format_date_time<boost::posix_time::time_duration>(
					boost::log::expressions::attr<boost::posix_time::time_duration>(name), it->second);
		}
		else {
			return boost::log::expressions::stream
				<< boost::log::expressions::attr<boost::posix_time::time_duration>(name);
		}
	}
};

void TRN::Helper::Logger::initialize(const std::string& configFileName) {
	// Disable all exceptions
	boost::log::core::get()->set_exception_handler(boost::log::make_exception_suppressor());

	// Add common attributes: LineID, TimeStamp, ProcessID, ThreadID
	boost::log::add_common_attributes();
	// Add boost log timer as global attribute Uptime
	boost::log::core::get()->add_global_attribute("Uptime", boost::log::attributes::timer());
	// Allows %Severity% to be used in ini config file for property Filter.
	boost::log::register_simple_filter_factory<boost::log::trivial::severity_level, char>("Severity");
	// Allows %Severity% to be used in ini config file for property Format.
	boost::log::register_simple_formatter_factory<boost::log::trivial::severity_level, char>("Severity");
	// Allows %TimeStamp(format=\"%Y.%m.%d %H:%M:%S.%f\")% to be used in ini config file for property Format.
	boost::log::register_formatter_factory("TimeStamp", boost::make_shared<TimeStampFormatterFactory>());
	// Allows %Uptime(format=\"%O:%M:%S.%f\")% to be used in ini config file for property Format.
	boost::log::register_formatter_factory("Uptime", boost::make_shared<UptimeFormatterFactory>());

	if (configFileName.empty()) {
		// Make sure we log to console if nothing specified.
		// Severity logger logs to console by default.
	}
	else {
		std::ifstream ifs(configFileName);
		if (!ifs.is_open()) {
			// We can log this because console is setup by default
			LOG_WARN("Unable to open logging config file: " << configFileName);
		}
		else {
			try {
				// Still can throw even with the exception suppressor above.
				boost::log::init_from_stream(ifs);
			}
			catch (std::exception& e) {
				std::string err = "Caught exception initializing boost logging: ";
				err += e.what();
				// Since we cannot be sure of boost log state, output to cerr and cout.
				ERROR_LOGGER << "ERROR: " << err ;
				std::cout << "ERROR: " << err ;
				LOG_ERROR(err);
			}
		}
	}

	// Indicate start of logging
	LOG_INFO("Log Start");
}

void TRN::Helper::Logger::disable() {
	boost::log::core::get()->set_logging_enabled(false);
}

void TRN::Helper::Logger::addDataFileLog(const std::string& logFileName) {
	// Create a text file sink
	boost::shared_ptr<boost::log::sinks::text_ostream_backend> backend(
		new boost::log::sinks::text_ostream_backend());
	backend->add_stream(boost::shared_ptr<std::ostream>(new std::ofstream(logFileName)));

	// Flush after each log record
	backend->auto_flush(true);

	// Create a sink for the backend
	typedef boost::log::sinks::synchronous_sink<boost::log::sinks::text_ostream_backend> sink_t;
	boost::shared_ptr<sink_t> sink(new sink_t(backend));

	// The log output formatter
	sink->set_formatter(
		boost::log::expressions::format("[%1%][%2%] %3%")
		% boost::log::expressions::attr<boost::posix_time::ptime>("TimeStamp")
		% boost::log::trivial::severity
		% boost::log::expressions::smessage
	);

	// Filter by severity and by DATALF channel
	sink->set_filter(
		boost::log::trivial::severity >= boost::log::trivial::info &&
		boost::log::expressions::attr<std::string>("Channel") == "DATALF");

	// Add it to the core
	boost::log::core::get()->add_sink(sink);
}*/