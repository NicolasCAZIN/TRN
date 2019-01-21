#include <armadillo>

static const std::size_t SIZE = 1024;
static const std::size_t ROWS = SIZE;
static const std::size_t COLS = SIZE;
static const std::size_t PC = 256;

static void compute_firing_rate_map(arma::mat &firing_rate_map)
{
#pragma omp parallel for
	for (int pc = 0; pc < PC; pc)
	{
		for (std::size_t row = 0; row < ROWS; row++)
		{
			for (std::size_t col = 0; col < COLS; col++)
			{

			}
		}
	}
}

int main(int argc, char *argv[])
{


	arma::mat firing_rate_map(ROWS * COLS, PC);

	compute_firing_rate_map(firing_rate_map);
}