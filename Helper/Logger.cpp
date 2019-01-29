#include "stdafx.h"
#include "Logger.h"


static std::recursive_mutex mutex{};

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
	Color()
	{
		CONSOLE_SCREEN_BUFFER_INFO info;
		GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info);
		value = info.wAttributes;
	}
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
static const Color NEUTRAL_COLOR;

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
			os << TRACE_COLOR << "TRACE      " << NEUTRAL_COLOR;
			break;
		case TRN::Helper::Logger::Severity::DEBUG_LEVEL:
			os << DEBUG_COLOR <<  "DEBUG      " << NEUTRAL_COLOR;
			break;
		case TRN::Helper::Logger::Severity::INFORMATION_LEVEL:
			os << INFORMATION_COLOR << "INFORMATION" << NEUTRAL_COLOR;
			break;
		case TRN::Helper::Logger::Severity::WARNING_LEVEL:
			os << WARNING_COLOR << "WARNING    " << NEUTRAL_COLOR;
			break;
		case TRN::Helper::Logger::Severity::ERROR_LEVEL:
			os << ERROR_COLOR << "ERROR      " << NEUTRAL_COLOR;
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
static std::string filename;

static TRN::Helper::Logger::Severity level = TRN::Helper::Logger::Severity::INFORMATION_LEVEL;
static bool exit_on_error = true;

struct TRN::Helper::Logger::Handle
{
	TRN::Helper::Logger::Severity severity;
	std::string module;
	std::ostringstream stream;
};

TRN::Helper::Logger::Logger(const TRN::Helper::Logger::Severity &severity, const std::string &module) : handle(std::make_unique<Handle>())
{
	handle->severity = severity;
	handle->module = module;
	std::lock_guard<std::recursive_mutex> guard(mutex);
	if (hostname.empty())
		hostname = boost::asio::ip::host_name();
	if (filename.empty())
		filename = "trn_" + std::to_string(getpid()) + ".log";
}

TRN::Helper::Logger::~Logger()
{
	std::lock_guard<std::recursive_mutex> guard(mutex);
	if (handle->severity >= level)
	{
		auto now = boost::posix_time::microsec_clock::universal_time();
		auto _str = handle->stream.str();

		if (_str[_str.length()] != '.')
			_str = _str + '.';
		_str[0] = std::toupper(_str[0]);
		stream(handle->severity) << NEUTRAL_COLOR << handle->severity << NEUTRAL_COLOR << " " << DATE_COLOR << "[" << now << "]" << NEUTRAL_COLOR << " " << MODULE_COLOR << handle->module << "@" << hostname << NEUTRAL_COLOR << " -> " << MESSAGE_COLOR << _str << NEUTRAL_COLOR << std::endl << std::flush;
		try
		{
			int attempts = 0;
			bool written = false;
			do
			{
				std::ofstream fstream(filename, std::ofstream::out | std::ofstream::app);

				if (fstream.is_open())
				{
					//boost::interprocess::file_lock flock(filename.c_str());
					//boost::interprocess::scoped_lock<boost::interprocess::file_lock> e_lock(flock);
					fstream << handle->severity << " " << "[" << now << "]" << " " << handle->module << "@" << hostname << " -> " << _str << std::endl << std::flush;

					fstream.close();
					written = true;
				}
		
				attempts++;
			} while (!written && attempts <= 5);

			if (!written)
				throw std::runtime_error("Unable to write log file after 5 attempts");
		}
		catch (std::exception &e)
		{
			handle->severity = Severity::ERROR_LEVEL;
			stream(handle->severity) << handle->severity << NEUTRAL_COLOR << " " << DATE_COLOR << "[" << now << "]" << NEUTRAL_COLOR << " " << MODULE_COLOR << handle->module << "@" << hostname << NEUTRAL_COLOR << " -> " << MESSAGE_COLOR << e.what() << NEUTRAL_COLOR << std::endl << std::flush;

		}
	}
	if (exit_on_error && handle->severity == Severity::ERROR_LEVEL)
	{
		exit(EXIT_FAILURE);
	}
	handle.reset();

}

std::ostream &TRN::Helper::Logger::ostream()
{
	return handle->stream;
}

void TRN::Helper::Logger::setup(const TRN::Helper::Logger::Severity &severity, const bool &exit_on_error)
{
	std::lock_guard<std::recursive_mutex> guard(mutex);
	level = severity;
	::exit_on_error = exit_on_error;
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
