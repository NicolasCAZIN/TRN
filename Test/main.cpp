#include <iostream>
#include <chrono>
#include <thread>
#include <queue>
#include <mutex>
#include <ctime>
#include "vld.h"
#include "mat.h"
#include "Helper/Queue.h"

std::mutex mutex;
int current_version = 0;

void update(const char *filename, const int next_version,  mxArray *result)
{
	std::unique_lock<std::mutex> lock(mutex);
	
	if (next_version > current_version)
	{
		auto pmat = matOpen(filename, "w");
		matError error = matPutVariable(pmat, "result", result);
		error = matClose(pmat);

		//result = matGetVariable(pmat, "result");
	

		current_version = next_version;

		std::cout << "Version "<< current_version<< " SAVED" << std::endl;
	}

}

int main(int argc, char *argv[])
{
	const char * filename = "Z:\\Nicolas\\JNS_data\\recombination\\result.mat";
	matError error;
	auto pmat = matOpen(filename, "w");
	assert(pmat);


	const char *RESULT_FIELD_LABELS[] = { "f1", "f2" };
	const int RESULTS_FIELD_NUMBER = sizeof(RESULT_FIELD_LABELS) / sizeof(RESULT_FIELD_LABELS[0]);
	mxArray *result = mxCreateStructMatrix(1, 1, RESULTS_FIELD_NUMBER, RESULT_FIELD_LABELS);
	error = matPutVariable(pmat, "result", result);
	assert(error == 0);
	error = matClose(pmat);
	assert(error == 0);



	const char *F1_FIELD_LABELS[] = { "x1", "x2" };
	const int F1_FIELD_NUMBER = sizeof(F1_FIELD_LABELS) / sizeof(F1_FIELD_LABELS[0]);

	float TIMEOUT = 20.0f;
	auto t0 = std::clock();
	TRN::Helper::Queue<std::pair<int, mxArray *>>  to_save;

	std::thread dumper([&to_save, &filename]()
	{
		std::pair<int, mxArray *> deep_copy;
		while (to_save.dequeue(deep_copy))
		{
			update(filename, deep_copy.first, deep_copy.second);
			mxDestroyArray(deep_copy.second);
		}
	});

	const int K = 40000;
	auto begin = std::clock();
	for (int k = 1; k <= K; k++)
	{
		//std::cout << "k " << k << std::endl;
		mxArray *old_f1 = mxGetField(result, 0, "f1");

		mxArray *new_f1 = mxCreateStructMatrix(k, 1, F1_FIELD_NUMBER, F1_FIELD_LABELS);

		mxArray *x1 = mxCreateNumericMatrix(1, 1, mxClassID::mxINT32_CLASS, mxComplexity::mxREAL);
		assert(x1); *(int *)mxGetData(x1) = k;

		mxArray *x2 = mxCreateNumericMatrix(100, 100, mxClassID::mxINT32_CLASS, mxComplexity::mxREAL);
		assert(x2);

		mxSetField(new_f1, k - 1, "x1", x1);
		mxSetField(new_f1, k - 1, "x2", x2);
		//	mxDestroyArray(x1);
			//mxDestroyArray(x2);


		if (old_f1 != NULL)
		{
			auto N = mxGetNumberOfElements(old_f1);
			assert(N == k - 1);

			for (int n = 0; n < N; n++)
			{
				x1 = mxGetField(old_f1, n, "x1"); assert(x1);
				x2 = mxGetField(old_f1, n, "x2"); assert(x2);

				mxSetField(new_f1, n, "x1", x1);
				mxSetField(new_f1, n, "x2", x2);
				//mxDestroyArray(x1);
				//
			}
		
		}

		mxSetField(result, 0, "f1", new_f1);

		auto t1 = std::clock();
		auto elapsed = (t1 - t0) / (float)CLOCKS_PER_SEC;
		if (elapsed > TIMEOUT)
		{

			

			std::unique_lock<std::mutex> lock(mutex, std::defer_lock);
			if (lock.try_lock())
			{
				auto t0_ = std::clock();
				if (k > current_version)
				{
					std::cout << "REQUESTING version " << k << std::endl;
					to_save.enqueue(std::make_pair(k, mxDuplicateArray(result)));
				}

				auto t1_ = std::clock();
				auto elapsed = (t1_ - t0_) / (float)CLOCKS_PER_SEC;
				std::cout << elapsed << " save latency" << std::endl;
			}


			t0 = std::clock();
		}

	}

	to_save.invalidate();
	if (dumper.joinable())
		dumper.join();
	update(filename, K, result);
	mxDestroyArray(result);
	auto end = std::clock();
	auto elapsed = (end - begin) / (float)CLOCKS_PER_SEC;

	std::cout << "overall time " << elapsed << std::endl;
	return 0;
}