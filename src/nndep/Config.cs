using System;
using System.IO;
using nndep.Util;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;
using System.Text;

namespace nndep
{
	static class Global
	{
		public static readonly long TimeStamp = DateTime.Now.ToFileTime();
		public static Logger Logger;
        public static string Mark = TimeStamp.ToString();
	}

	enum OptType
	{
		sgd,
		adagrad,
		adam
	}

    [Serializable]
    class Config
	{
		public int Seed;
		public const string Root = "-ROOT-";

		public const string Separator = "###################";

		[JsonConverter(typeof(StringEnumConverter))] public readonly OptType OptType;

		public float ClipBound;
		public float DecayRate;
		public string DevFile;

		public int EmbeddingSize;
        public string EmbedFile;
        public float Eps;
        public int HiddenSize;
        public float InitRange;
        public int IterPerEpoch;
        public float L2RegFactor;
        public float LearningRate;
        public int MaxEpoch;
        public int MinibatchSize;
        public string TestFile;

		public string TrainFile;
        [JsonConverter(typeof(StringEnumConverter))] public ArcStd.ArcStandardConfig.OracleType OraType;

		public int WordCutOff;
        public bool UseCpos;
        public string Model;
        public RunMode Mode;

        public enum RunMode { train, dev, test}

		public Config()
		{
			WordCutOff = 1;
			InitRange = 0.01f;
			MinibatchSize = 1;
			MaxEpoch = 20;
			IterPerEpoch = 1;
			LearningRate = 0.001f;
			DecayRate = 0.09f;
			Eps = 1e-8f;
			L2RegFactor = 1e-8f;
			ClipBound = 5;
			HiddenSize = 200;
			EmbeddingSize = 50;
		    Seed = 12976;
            UseCpos = false;
			TrainFile = null;
			DevFile = null;
			TestFile = null;
			EmbedFile = null;
			OptType = OptType.adam;
            OraType = ArcStd.ArcStandardConfig.OracleType.standard;
            Model = null;
            Mode = RunMode.train;
		}


		public void PrintParameters()
		{
            Global.Logger.WriteLine($"RunMode = {Mode}");
            Global.Logger.WriteLine($"Mark = {Global.Mark}");
			Global.Logger.WriteLine($"Timestamp = {Global.TimeStamp}");
			Global.Logger.WriteLine($"Train File = {TrainFile}");
			Global.Logger.WriteLine($"Dev File = {DevFile}");
			Global.Logger.WriteLine($"Test File = {TestFile}");
			Global.Logger.WriteLine($"Embedding File = {EmbedFile}");


		    Global.Logger.WriteLine($"RandomSeed = {Seed}");
            Global.Logger.WriteLine($"MaxEpoch = {MaxEpoch}");
			Global.Logger.WriteLine($"IterPerEpoch = {IterPerEpoch}");
            Global.Logger.WriteLine($"UseCPOS = {UseCpos}");
            Global.Logger.WriteLine($"OracleType = {OraType}");
            Global.Logger.WriteLine($"MinibatchSize = {MinibatchSize:g2}");



			Global.Logger.WriteLine($"HiddenSize = {HiddenSize}");
			Global.Logger.WriteLine($"EmbeddingSize = {EmbeddingSize}");
			Global.Logger.WriteLine($"WordCutOff = {WordCutOff}");
			Global.Logger.WriteLine($"InitRange = {InitRange:g2}");

			Global.Logger.WriteLine($"OptType = {OptType}");
			Global.Logger.WriteLine($"LearningRate = {LearningRate:g2}");
			Global.Logger.WriteLine($"DecayRate = {DecayRate:g2}");
			Global.Logger.WriteLine($"Eps = {Eps:g2}");
			Global.Logger.WriteLine($"RegParameter = {L2RegFactor:g2}");
			Global.Logger.WriteLine($"ClipBound = {ClipBound:g2}");
		}

		public static Config ReadFromJson(string file)
		{
			var conf = JsonConvert.DeserializeObject<Config>(File.ReadAllText(file, Encoding.UTF8));
			return conf;
		}

        public void WriteToJson(string file)
        {
            File.WriteAllText(file, JsonConvert.SerializeObject(this, Formatting.Indented), Encoding.UTF8);
        }
	}
}