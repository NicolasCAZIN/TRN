import java.util.List;
import java.util.ArrayList;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


import TRN4JAVA.Engine;
import TRN4JAVA.Engine.Backend.Local;
import TRN4JAVA.Simulation;

public class Test
{
	static
	{
		System.loadLibrary("vcomp120");
			System.loadLibrary("msvcr120");
			System.loadLibrary("msvcp120");
	

				System.loadLibrary("vcruntime140");
			System.loadLibrary("concrt140");
				System.loadLibrary("msvcp140");

	
			System.loadLibrary("vcomp140");
	
		
			System.loadLibrary("tbb");
			System.loadLibrary("tbbmalloc");

			System.loadLibrary("libimalloc");
			System.loadLibrary("libiomp5md");
			System.loadLibrary("mkl_core");
			System.loadLibrary("mkl_intel_thread");
			System.loadLibrary("mkl_sequential");
			System.loadLibrary("mkl_tbb_thread");
			System.loadLibrary("mkl_def");
			System.loadLibrary("mkl_avx");
			System.loadLibrary("mkl_avx2");
			System.loadLibrary("mkl_avx512");
			System.loadLibrary("mkl_avx512_mic");

			System.loadLibrary("mkl_mc");
			System.loadLibrary("mkl_mc3");
			System.loadLibrary("mkl_rt");

			System.loadLibrary("mkl_vml_avx");
			System.loadLibrary("mkl_vml_avx2");
			System.loadLibrary("mkl_vml_avx512");
			System.loadLibrary("mkl_vml_avx512_mic");
			System.loadLibrary("mkl_vml_cmpt");
			System.loadLibrary("mkl_vml_def");
			System.loadLibrary("mkl_vml_mc");
			System.loadLibrary("mkl_vml_mc2");
			System.loadLibrary("mkl_vml_mc3");
	




			System.loadLibrary("hdf5");

			System.loadLibrary("icudt56");
			System.loadLibrary("icuin56");
			System.loadLibrary("icuio56");
			System.loadLibrary("icuuc56");
			System.loadLibrary("zlib1");
			System.loadLibrary("libexpat");

	
			System.loadLibrary("libmwfl");
			System.loadLibrary("libmwfoundation_usm");
			System.loadLibrary("libmwi18n");
			System.loadLibrary("libmwresource_core");
	
			System.loadLibrary("libut");
			System.loadLibrary("libmat");
			System.loadLibrary("libmx");




			System.loadLibrary("cudart64_90");
			System.loadLibrary("cublas64_90");
			System.loadLibrary("curand64_90");

			System.loadLibrary("boost_chrono-vc120-mt-1_56");
			System.loadLibrary("boost_date_time-vc120-mt-1_56");
			System.loadLibrary("boost_filesystem-vc120-mt-1_56");
			System.loadLibrary("boost_log-vc120-mt-1_56");
			System.loadLibrary("boost_regex-vc120-mt-1_56");
			System.loadLibrary("boost_serialization-vc120-mt-1_56");
			System.loadLibrary("boost_signals-vc120-mt-1_56");
			System.loadLibrary("boost_thread-vc120-mt-1_56");
			System.loadLibrary("boost_system-vc120-mt-1_56");

	
			System.loadLibrary("boost_filesystem-vc140-mt-1_62");
			System.loadLibrary("boost_iostreams-vc140-mt-1_62");
			System.loadLibrary("boost_mpi-vc140-mt-1_62");
			System.loadLibrary("boost_program_options-vc140-mt-1_62");
			System.loadLibrary("boost_serialization-vc140-mt-1_62");
			System.loadLibrary("boost_system-vc140-mt-1_62");
			System.loadLibrary("boost_zlib-vc140-mt-1_62");
			System.loadLibrary("boost_bzip2-vc140-mt-1_62");

			System.loadLibrary("Backend");
			System.loadLibrary("GPU");
			System.loadLibrary("CPU");

			System.loadLibrary("Helper");
			System.loadLibrary("Core");
			System.loadLibrary("Initializer");
			System.loadLibrary("Loop");
			System.loadLibrary("Measurement");
			System.loadLibrary("Mutator");
			System.loadLibrary("Reservoir");
			System.loadLibrary("Scheduler");
			System.loadLibrary("Simulator");
			System.loadLibrary("Model");

			System.loadLibrary("Network");

			System.loadLibrary("Engine");
			System.loadLibrary("Remote");
			System.loadLibrary("Distributed");
			System.loadLibrary("Local");
		
			System.loadLibrary("ViewModel");
			System.loadLibrary("TRN4CPP");
			System.loadLibrary("TRN4JAVA");
	}
	
	public static class PlaceCell
	{
		private final static float THRESHOLD = 0.2f;
		
		private final float K;
		private final float x;
		private final float y;
		
		public PlaceCell(float x, float y, float radius)
		{
			this.x = x;
			this.y = y;
			this.K = (float)Math.log(THRESHOLD) / (radius * radius);
			
	
		}
		
		public float activation(float x, float y)
		{
			float dx = x - this.x;
			float dy = y - this.y;
			
			return (float)Math.exp((dx*dx + dy*dy) * K);
		}
	}
	
	public static class Rat 
	{
		public static class Position extends TRN4JAVA.Simulation.Loop
		{
			private Rat rat;
			public Position(Rat rat)
			{
				this.rat = rat;
			}
			@Override
			public void callback(final long id, final long trial, final long evaluation, final float prediction[], final long rows, final long cols)
			{
				new Thread(new Runnable() {
           public void run() {
               			rat.move(id, trial, evaluation, prediction, rows, cols);          
    }
}).start();
	
			}
		};
		
		public static class Stimulus extends TRN4JAVA.Simulation.Loop
		{
			private Rat rat;
			public Stimulus(Rat rat)
			{
				this.rat = rat;
			}
			@Override
			public void callback(final long id, final long trial, final long evaluation, final float prediction[], final long rows, final long cols)
			{

			}
		};
		
		public final Position position;
		public final Stimulus stimulus;
		private final List<PlaceCell> place_cells;
	
		public Rat(final List<PlaceCell> place_cells)
		{
			position = new Position(this);
			stimulus = new Stimulus(this);
			this.place_cells = place_cells;
		}
		
		protected void move(final long id, final long trial, final long evaluation, final float location[], final long rows, final long cols)
		{
			assert(cols == 2);
			float activation[] = new float[(int)rows * place_cells.size()];
			for (int batch = 0; batch < rows; batch++)
			{
				for (int place_cell = 0; place_cell < place_cells.size(); place_cell++)
				{
					float x = location[batch * 2 + 0];
					float y = location[batch * 2 + 1];
					activation[batch * place_cells.size() + place_cell] = place_cells.get(place_cell).activation(x, y);
					
				
	
				}
			}
		
			position.notify(id, trial, evaluation, location, rows, cols);
			stimulus.notify(id, trial, evaluation, activation, rows, place_cells.size());
		}
	};
	
	public final static void main(String args[])
	{
		try
		{
			if (args.length != 2)
				throw new java.lang.RuntimeException("experiment and placecells filename must me provided");
			String experiment = args[0];
			String placecells = args[1];
			
			BufferedReader br = new BufferedReader(new FileReader(placecells));
			String line = br.readLine();
			List<PlaceCell> place_cells = new ArrayList<PlaceCell>();
			
			while ((line = br.readLine()) != null) 
			{
				String[] tokens = line.split("\t");
	
				float x = Float.parseFloat(tokens[3]);
				float y = Float.parseFloat(tokens[4]);
				float radius = Float.parseFloat(tokens[5]);
				place_cells.add(new PlaceCell(x, y, radius));
			}
		
			Rat rat = new Rat(place_cells);
			int indices[]={1};
			
			TRN4JAVA.Engine.Backend.Local.initialize(indices);

			TRN4JAVA.Simulation.Loop.Position.install(rat.position);	
			TRN4JAVA.Simulation.Loop.Stimulus.install(rat.stimulus);	
			TRN4JAVA.Simulation.compute(experiment);
			
			TRN4JAVA.Engine.uninitialize();
		}
		catch (Exception e)
		{
			System.err.println("toto" + e);
		}
	}
	
}
	

