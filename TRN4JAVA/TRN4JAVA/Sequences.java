package TRN4JAVA;

public class Sequences
{
	public static class Plugin
	{
		public static native void initialize(final String library_path, final String name, final java.util.Map<String, String> arguments);
	}

	public static native void	declare(final String label, final float sequence[], final long rows, final long cols, final String tag);
}