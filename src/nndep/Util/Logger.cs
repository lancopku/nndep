using System;
using System.IO;
using System.Text;

namespace nndep.Util
{
	public class Logger
	{
	    public Logger()
	    {
	        Output += (sender, args) => { Console.Write(args.Format); };
	        OutputConsole += (sender, args) => { Console.Write(args.Format); };
	        OutputLine += (sender, args) => { Console.WriteLine(args.Format); };

	        var filename = $"{Global.Mark}\\log-{Global.Mark}.txt";
	        OutputLine +=
	            (sender, args) =>
	            {
	                File.AppendAllText(filename, args.Format + Environment.NewLine,
	                    Encoding.UTF8);
	            };
	        Output +=
	            (sender, args) =>
	            {
	                File.AppendAllText(filename, args.Format,
	                    Encoding.UTF8);
	            };

	    }

	    protected virtual void OnWrite(OutputEventArgs e)
		{
			Output?.Invoke(this, e);
		}

		protected virtual void OnWriteConsole(OutputEventArgs e)
		{
			OutputConsole?.Invoke(this, e);
		}

		protected virtual void OnWriteLine(OutputEventArgs e)
		{
			OutputLine?.Invoke(this, e);
		}

		private event OutputEventHandler Output;
		private event OutputConsoleEventHandler OutputConsole;
		private event OutputLineEventHandler OutputLine;

		public void Write(string format)
		{
			OnWrite(new OutputEventArgs(format));
		}

		public void WriteConsole(string format)
		{
			OnWriteConsole(new OutputEventArgs(format));
		}

		public void WriteLine(string format)
		{
			OnWriteLine(new OutputEventArgs(format));
		}

		public void WriteLine()
		{
			OnWriteLine(new OutputEventArgs(""));
		}

		private delegate void OutputEventHandler(object sender, OutputEventArgs e);

		private delegate void OutputConsoleEventHandler(object sender, OutputEventArgs e);

		private delegate void OutputLineEventHandler(object sender, OutputEventArgs e);

		protected class OutputEventArgs : EventArgs
		{
			public readonly string Format;

			public OutputEventArgs(string format)
			{
				Format = format;
			}
		}
	}
}