#include <QtCore/QCoreApplication>

#include <cuda_runtime_api.h>
#include <map>
#include <iostream>
#include <memory>
#include <functional>
/*#include <boost/signals2.hpp>
#include <boost/bind.hpp>
#include <boost/lexical_cast.hpp>*/


#include "Helper/Visitor.h"
#include "Helper/Observer.h"

#include "GPU/Memory.h"
#include "GPU/Random.h"

#define TASKS 10000

#define STIMULUS_SIZE 256
#define READOUT_SIZE 256
#define OBSERVATIONS 132

#define EPOCHS 1000

class View
{
public:
	void on_results_received(std::string results) 
	{
		std::cout << "received results : " << results << std::endl;
	}
	void on_performances_received(float performances) 
	{
		std::cout << "received performances : " << performances << std::endl;
	}
};


namespace TRN
{
	namespace Data
	{

		struct Distribution
		{
			enum Type
			{
				GAUSSIAN,
				UNIFORM,
				CUSTOM
			};
			Type type;

			template <typename T, int I>
			struct Arguments {};

			template<typename T, TRN::Data::Distribution::Type::GAUSSIAN>
			struct Arguments
			{
				T mean;
				T stddev;
			};
			template<typename T, UNIFORM>
			struct Arguments
			{
				T a;
				T b;
				T sparsity;
			};
			template<typename T, CUSTOM>
			{
				int rows;
				int cols;
				std::vector<T> samples;
			};
	
		};




		/*template<class T, int I>  // primary template
		struct A {
			void f(); // member declaration
		};

		template<class T, int I>
		void A<T, I>::f() { } // primary template member definition

							  // partial specialization
		template<class T>
		struct A<T, 2> {
			void f();
			void g();
			void h();
		};

		// member of partial specialization
		template<class T>
		void A<T, 2>::g() { }

		// explicit (full) specialization
		// of a member of partial specialization
		template<>
		void A<char, 2>::h() {}*/

		struct Parameters
		{
			struct Reservoir
			{
				int reservoir_size;
				int readout_size;
				int simulus_size;
				Distribution feedforward;
				Distribution recurrent;
				Distribution feedback;
				Distribution readout;
			};
			Reservoir reservoir;

			struct Loop
			{
				enum Type
				{
					COPY,
					SPATIAL_FILTER,
					CUSTOM
				};
				Type type;
			};
			Loop loop;

			struct Scheduler
			{
				enum Type
				{
					TILED,
					SNIPPETS,
					CUSTOM
				};
				int epochs;

			};
		};
		struct Results
		{
			float mse;
			float frechet;
		};
		struct States
		{
			int rows;
			int cols;
			std::vector<float> samples;
		};
		struct Scheduling
		{
			std::vector<int> offsets;
			std::vector<int> durations;
			int epochs;
		};
		struct Dataset
		{
			States incoming;
			States expected;
			Scheduling scheduling;
		};
	};
	





	/*namespace CPU
	{
		class Memory : public TRN::Backend::Memory
		{
		public :
			virtual void allocate(void **ptr, std::size_t &stride, const std::size_t &depth, const std::size_t &width, const std::size_t &height) override
			{
				stride = width;
				*ptr = std::malloc(depth * width * height);
			}

			virtual void deallocate(void *ptr) override
			{
				std::free(ptr);
			}

			virtual void blank(void *ptr, const std::size_t &size) override
			{
				std::memset(ptr, 0, size);
			}
		};

		class Random : public TRN::Backend::Random
		{
		public :
			virtual void uniform(float *ptr, std::size_t elements, const float &a, const float &b, const float &sparsity = 1.0f)
			{

			}
			virtual void gaussian(float *ptr, std::size_t elements, const float &mu, const float &sigma)
			{

			}
		};
	};
	namespace GPU
	{
		class Context
		{
		public :
			Context(int index)
			{
				check
			}

		private :
			mutable int gpu;
			mutable std::string name;
			mutable cudaStream_t stream;
		};

		class Memory : public TRN::Backend::Memory
		{
		public :
			Memory(int index)
		public:
			virtual void allocate(void **ptr, std::size_t &stride, const std::size_t &depth, const std::size_t &width, const std::size_t &height) override
			{
				stride = width;
				*ptr = std::malloc(depth * width * height);
			}

			virtual void deallocate(void *ptr) override
			{
				std::free(ptr);
			}

			virtual void blank(void *ptr, const std::size_t &size) override
			{
				std::memset(ptr, 0, size);
			}
		};

		class Random : public TRN::Backend::Random
		{
		public:
			virtual void uniform(float *ptr, std::size_t elements, const float &a, const float &b, const float &sparsity = 1.0f)
			{

			}
			virtual void gaussian(float *ptr, std::size_t elements, const float &mu, const float &sigma)
			{

			}
		};
	};
	void initialize()
	{
		TRN::Helper::Register<TRN::Backend::Memory, TRN::CPU::Memory>("CPU");
		TRN::Helper::Register<TRN::Backend::Random, TRN::CPU::Random>("CPU");
		TRN::Helper::Register<TRN::Backend::Memory, TRN::GPU::Memory, int>("GPU");
		TRN::Helper::Register<TRN::Backend::Random, TRN::GPU::Random, int>("GPU");
	}*/
	/*
	namespace Engine
	{
		class Configurable
		{
		protected:
			float c;

		public:
			Configurable(float c = 50.0f) : c(c)
			{

			}

		public:
			virtual void set_c(float c)
			{
				this->c = c;
			}
		};

		class Matrix : public TRN::Helper::Bridge<
		{
		private :
			mutable int rows;
			mutable int cols;
			mutable int stride;


		};


		class Initializer //public TRN::Helper::Visitor<TRN::Engine::Matrix>
		{
		public :
			virtual void initialize() = 0;
		};

		class Gaussian : public TRN::Helper::Bridge<TRN::Engine::Initiliazer, TRN::Backend>

		class Model : public TRN::Helper::Visitor<Data::Parameters>,
					  public TRN::Helper::Visitor<Data::Results>,
					public TRN::Helper::Visitor<Data::States>
		{
		protected :
			Model(float c) : TRN::Engine::Configurable(c) {}

		public:
			virtual float compute() = 0;
		};

		

			class ModelBuilder : public TRN::Engine::ModelBuilder
			{
			private:
				int index;

			public:
				ModelBuilder(int index) : index(index) {}

			public:
				virtual  std::shared_ptr<TRN::Engine::Model> build() override
				{
					return std::make_shared<TRN::Engine::GPUModel>(c);
				}
			};
		}


	namespace CPU
	{
		class Model : public TRN::Engine::Model
		{
		public:
			Model(float c) : TRN::Engine::Model(c) {}

		public:
			virtual float compute() override
			{
				std::cout << "CPU is used" << std::endl;
				return(c + 10.0f);
			}

			virtual void visit(std::shared_ptr<TRN::Data::States> states) override
			{
				states->rows = 3;
				states->cols = 10;
				states->samples.resize(states->rows * states->cols);
				std::fill(states->samples.begin(), states->samples.end(), 1.0f);
			}

			virtual void visit(std::shared_ptr<TRN::Data::Results> results) override
			{
				results->frechet = 0.1f;
				results->mse = 0.01f;
			}
		};

		class ModelBuilder : public TRN::Engine::ModelBuilder
		{
		public:
			virtual  std::shared_ptr<TRN::Engine::Model> build() override
			{
				return std::make_shared<TRN::Engine::CPU::Model>(c);
			}
		};
	}

	namespace GPU
	{
		class Model : public TRN::Engine::Model
		{
		private:
			int index;
		public:
			Model(int index, float c) : TRN::Engine::Model(c), index(index)
			{
			}

		public:
			virtual float compute() override
			{
				std::cout << "GPU #" << index << "is used" << std::endl;
				return c + 100.0f;
			}

			virtual void visit(std::shared_ptr<TRN::Data::States> states) override
			{
				states->rows = 3;
				states->cols = 10;
				states->samples.resize(states->rows * states->cols);
				std::fill(states->samples.begin(), states->samples.end(), 10.0f);
			}

			virtual void visit(std::shared_ptr<TRN::Data::Results> results) override
			{
				results->frechet = 0.01f;
				results->mse = 0.001f;
			}
		};

		class ModelBuilder : public Helper::Builder<TRN::Engine::Model>, public TRN::Engine::Configurable
		{
		};

		class ModelBuilderFactory
		{
		public:
			static std::shared_ptr<ModelBuilder> create(int index)
			{
				switch (index)
				{
				case 0:
					return std::make_shared<CPUModelBuilder>();
				default:
					return std::make_shared<GPUModelBuilder>(index);
				}
			}
		};

		class Simulator : public TRN::Engine::Configurable
		{
		public:
			virtual void launch() = 0;
		};

		class LocalSimulator : public Simulator
		{
		public:
			LocalSimulator(int index, std::function<void(const std::string &)> functor) :
				builder(ModelBuilderFactory::create(index)),
				functor(functor)
			{
				target = builder;
			}

		private:
			std::shared_ptr<ModelBuilder> builder;
			std::shared_ptr<Model> model;
			std::shared_ptr<Configurable> target;
			std::function<void(std::string &)> functor;

		public:
			virtual void launch() override
			{
				if (!model)
				{
					model = builder->build();
					target = model;
				}
				auto res = model->compute();
				std::string str = "toto # " + boost::lexical_cast<std::string>(res);
				functor(str);
			}

		public:
			virtual void set_c(float c) override
			{
				target->set_c(c);
			}


		};



		class ResultsSimulator : public Decorator<Simulator>
		{
		private:
			std::function<void(const std::string &)> functor;

		public:
			ResultsSimulator(std::shared_ptr<Simulator> simulator, std::function<void(const std::string &)> functor) :
				Decorator(simulator),
				functor(functor)
			{
			}

		public:
			virtual void launch() override
			{

				decorated->launch();
				float t1 = 1.0f;
				functor(str);
			}

		};

		class PerformancesSimulator : public Decorator<Simulator>
		{
		private:
			std::function<void(const float &)> functor;

		public:
			PerformancesSimulator(std::shared_ptr<Simulator> simulator, std::function<void(const float &)> functor) :
				Decorator(simulator),
				functor(functor)
			{
			}

		public:
			virtual void launch() override
			{
				float t0 = 0.0f;
				decorated->launch();
				float t1 = 1.0f;
				functor(t1 - t0);
			}
		};

	};


	


	class ModelView
	{
	private:
		std::map<int, std::shared_ptr<TRN::Simulator>> simulator;
		
	public :
		template<typename ... Args>
		void create_simulator(std::string type, Args &&...args)
		{
			if (type == "Local")
				simulator = std::make_shared<TRN:::LocalSimulator>(std::forward(args)...);

		}




	};
*/
	namespace Model
	{
		class Reservoir
		{

		};

		class Loop
		{

		};

		class Simulator : 
			public TRN::Helper::Observer<TRN::Data::Parameters::Reservoir>,
			public TRN::Helper::Observer<TRN::Data::Parameters::Loop>
		{
		private :
			std::shared_ptr<TRN::Backend::Memory> memory;
			std::shared_ptr<TRN::Backend::Random> random;

		private :
			std::shared_ptr<TRN::Model::Reservoir> reservoir;
			std::shared_ptr<TRN::Model::Loop> loop;

		public :
			virtual void update(const std::shared_ptr<TRN::Data::Parameters::Reservoir> parameters) override
			{
				reservoir = Model::Facade::create_reservoir(parameters);
			}
			virtual void update(const std::shared_ptr<TRN::Data::Parameters::Loop> parameters) override
			{
				loop = Model::Facade::create_loop(parameters);
			}
		};
	};

	namespace ModelView
	{
		class Engine : 
			public TRN::Helper::Observer<TRN::Data::Parameters::Reservoir>, 
			public TRN::Helper::Observer<TRN::Data::Parameters::Loop>,
			public TRN::Helper::Observable<TRN::Data::Parameters::Reservoir>
			public TRN::Helper::Observable<TRN::Data::Parameters::Loop>
		{
		public :
			virtual void update(const std::shared_ptr<TRN::Data::Parameters::Reservoir> parameters) override
			{
				notify(parameters);
			}
			virtual void update(const std::shared_ptr<TRN::Data::Parameters::Loop> parameters) override
			{
				notify(parameters);
			}
		};

	};

	namespace Local
	{
		class Engine : 
			public ModelView::Engine


		{
		private :
	

		};
	}

	namespace Remote
	{
		class Engine : public ModelView::Engine
		{
		};
	}
};




int main(int argc, char *argv[=])
{
	auto context = std::make_shared<TRN::GPU::Context>(3);

	auto memory = std::make_shared<TRN::GPU::Memory>(context);
	auto random = std::make_shared<TRN::GPU::Random>(context, 0);
	/*auto view = std::make_shared <View>();
	std::shared_ptr<TRN::ModelView> std::make_shared<TRN::LocalModelView>(functor);
//model_view = std::make_shared<TRN::PerformancesModelView>(model_view, [view](const float &performances) { view->on_performances_received(performances); });
	[view](const std::string &results) { view->on_results_received(results); }

	model_view->launch();*/
	//std::string str("toto");
	//handle(str);

	/*TRN::initialize();

	auto simulator = TRN::create_simulator("Local", 0);
	auto parameters = TRN::create_parameters();

	std::vector<float> incoming(STIMULUS_SIZE * OBSERVATIONS);
	std::vector<float> expected(READOUT_SIZE * OBSERVATIONS);

	auto stimulus = TRN::create_states(incoming, STIMULUS_SIZE, OBSERVATIONS);
	auto expected = TRN::create_states(expected, READOUT_SIZE, OBSERVATIONS);

	auto train = TRN::create_dataset(stimulus, expected, Data::Scheduling::TILED, EPOCHS);
	auto test = TRN::create_dataset(expected, expected, Data::Scheduling::TILED, 1);
	
	parameters->set_initial_state_scale(0.1f);
	parameters->set_leak_rate(0.1f);
	parameters->set_learning_rate(0.0001f);
	parameters->set_reservoir_size(2048);


	simulator->configure(parameters);
	simulator->configure(train, test);
	//simulator->set_train_dataset();
	simulator->train();*/
	//simulator->set_train_dataset();

	//auto parameters = TRN::load_parameters("");
	/*simulator->set_readout_size(256/2);
	simulator->set_reservoir_size(1024);
	simulator->set_stimulus_size(256);*/

	return (0);

	/*
		users land

		int trials = 100;
		std::vector<float> incoming;
		std::vector<float> expected;

#pragma omp parallel for
		for (int k = 0; k < K; k++)
		{
			//Frontend::Builder builder = Frontend::Factory::create("Remote", ip, port)
			

			simulator[k].set_id(k);
			simulator[k].set_reservoir_size(1024);
			simulator[k].set_recurrent_initializer_type(Frontend::Initializer::Type::SparseUniform);
			simulator[k].set_recurrent_initializer_seed(0);
			simulator[k].set_instances_number(10);
			simulator[k].set_recurrent_initializer_parameters({-1, 1, 0.7});


			std::vector<Report::States> train_states(trials);
			std::vector<Report::Performances> train_performances(trials);
			std::vector<Report::States> test_states(trials);
			std::vector<Report::Performances> test_performances(trials);
			std::vector<Results::AbsoluteMeanSquareError> amse(trials);
			std::vector<Results::RelativeMeanSquareError> rmse(trials);
			std::vector<Frontend::Simulator> simulators;

			// implicit command queue in simulator
			simulators[k].set_train_dataset(incoming, expected, offsets, durations, epochs);
			for (int trial = 0; trial < trials; trial++)
			{
				simulator[k].train(train_states[trial], train_performances[trial]);
				simulator[k].test(test_results[trial], test_states[trial], test_performances[trial]);
			}
		}
		
#pragma omp parallel for
		for (int k = 0; k < K; k++)
		{
			simulator[k].wait();
		}


	*/


	/*
	
		
	*/
	/*int variables = 256;
	int observations = 130;

	std::vector<float> samples(variables*observations);
	for (int k = 0; k < variables*observations; k++)
		samples[k] = k;

	int epochs = 1000;

	int gpus;
	checkCudaErrors(cudaGetDeviceCount(&gpus));


	


	auto reservoir = std::shared_ptr<CPU::Reservoir>::create(variables, variables, 1024);
	auto incoming = std::shared_ptr<CPU::Sequence>::create(samples.data(), observations - 1, variables);
	auto expected = std::shared_ptr<CPU::Sequence>::create(samples.data() + variables, observations - 1, variables);
	auto tiled = std::shared_ptr<CPU::Tiled>::create(incoming, expected);
	auto dataset = tiled->generate(epochs);
	
	auto instrumented = std::shared_ptr<Instrumented::Performances>::create(reservoir);
	instrumented->train(dataset);

	auto parameters = std::shared_ptr<Data::Parameters>::create();

	auto performances = std::shared_ptr<Data::Performances>::create();
	auto parameters_visitable = std::shared_ptr<Report::Visitable<Data::Parameters>>::create(parameters);
	auto performances_visitable = std::shared_ptr<Report::Visitable<Data::Performances>>::create(performances);
	parameters_visitable->accept(instrumented);
	performances_visitable->accept(instrumented);



	auto throughput = performances->get_cycles_per_second();

	qDebug() << "throughput : " << throughput << " Hz";


	return 0;
	//return a.exec();*/
}
