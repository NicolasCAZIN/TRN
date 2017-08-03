#include "stdafx.h"

PrintThread::~PrintThread()
{
	std::lock_guard<std::mutex> guard(_mutexPrint);
	std::cout << this->str();
}
std::mutex PrintThread::_mutexPrint{};
