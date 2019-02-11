#pragma once

class Server
{
private :
	class Handle;
	std::unique_ptr<Handle> handle;
};
