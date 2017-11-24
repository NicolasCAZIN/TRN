#include "stdafx.h"
#include "Messages.h"

void   TRN::Engine::encode(const unsigned short &number, const unsigned short &condition_number, const unsigned int &simulation_number, unsigned long long &id)
{
	Identifier identifier;

	identifier.number = number;
	identifier.condition_number = condition_number;
	identifier.simulation_number = simulation_number;
	id = identifier.id;
}

void TRN::Engine::decode(const unsigned long long &id, unsigned short &number, unsigned short &condition_number, unsigned int &simulation_number)
{
	Identifier identifier;

	identifier.id = id;
	number = identifier.number;
	condition_number = identifier.condition_number;
	simulation_number = identifier.simulation_number;
}