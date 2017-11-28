using System;
using System.IO;
using nndep.ArcStd;
using nndep.Util;

namespace nndep
{
	static class Program
	{
		static void Main(string[] args)
		{
			Config conf = null;
		    string name = null;
			if (args.Length > 0)
			{
				var path = args[0];
			    if (File.Exists(path))
			    {
			        conf = Config.ReadFromJson(path);
			        name = Path.GetFileNameWithoutExtension(path)?.Split('.')[0];
			    }
			    else
			    {
                    Console.WriteLine("Invalid Config");
			        Console.ReadKey();
                    return;
			    }
			}
			else if (File.Exists("default.json"))
            {
                conf = Config.ReadFromJson("default.json");
                name = "default";
            }
            else
			{
				conf = new Config();
			    name = "default";
			    conf.WriteToJson("default.json");
                Console.WriteLine("No Config Given. Template written to default.json");
			    Console.ReadKey();
			    return;
            }

            Global.Mark = conf.Mode == Config.RunMode.train ? $"{name}-{conf.EmbeddingSize}-{conf.HiddenSize}-{conf.OraType}-{Global.TimeStamp}" : $"{name}-{Path.GetFileNameWithoutExtension(conf.Model)}-{(conf.Mode == Config.RunMode.dev? Path.GetFileNameWithoutExtension(conf.DevFile):Path.GetFileNameWithoutExtension(conf.TestFile))}-{Global.TimeStamp}";
            Directory.CreateDirectory(Global.Mark);
			Global.Logger = new Logger();
			var depParser = new ArcstdBiLstm(conf);
		    if (!depParser.Status)
		    {

		        Console.WriteLine("Parser Init Failed.");
		        Console.ReadKey();
                return;
		    }
            switch (conf.Mode)
            {
                case Config.RunMode.train:
                    depParser.Train();
                    break;
                case Config.RunMode.dev:
                    depParser.Dev();
                    break;
                case Config.RunMode.test:
                    depParser.Tst();
                    break;
            }

			Console.ReadKey();
        }
    }
}