using System;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using nndep.Data;
using nndep.Util;

namespace nndep.ArcStd
{
    class ArcstdBiLstm
    {
        private Config _conf;
        private DataSet _dev;
        private MLP _model;
        private DataSet _test;
        private DataSet _train;
        public bool Status;

        public ArcstdBiLstm(Config config)
        {
            _conf = config;
            Global.Logger.WriteLine(Config.Separator);
            config.PrintParameters();


            
            if (_conf.Mode == Config.RunMode.train && (!File.Exists(config.TrainFile) || !File.Exists(config.DevFile)))
            {
                Console.WriteLine(Config.Separator);
                Console.WriteLine("train or dev file not found");
                Status = false;
                //throw new FileNotFoundException("train or dev file not found");
                return;
            }


            _train = new DataSet(config.TrainFile, config.UseCpos, new RandomNumberGenerator(config.Seed));
            _train.PrintTreeStats();

            _dev = new DataSet(config.DevFile, config.UseCpos);
            _dev.PrintTreeStats(false);

            if (File.Exists(config.TestFile))
            {
                _test = new DataSet(config.TestFile, config.UseCpos);
                _test.PrintTreeStats(false);
            }
            else
            {
                _test = null;
            }

            _train.GenerateMappings(config.WordCutOff);
            if (_conf.Model != null && File.Exists(_conf.Model))
            {
                Load(_conf.Model);
            }
            else
            {
                _model = new MLP(_conf, _train);
            }
            Status = true;

        }

        private void Load(string file)
        {
            try
            {
                var formatter = new BinaryFormatter();
                Stream stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.Read);
                _model = (MLP)formatter.Deserialize(stream);
                _model.Recover(_conf, _train);
                stream.Close();
            }
            catch
            {
                Console.WriteLine($"Model file doesn't exist, create a new model instead");
                _model = new MLP(_conf, _train);
            }

        }

        private void Save(string file)
        {
            var formatter = new BinaryFormatter();
            Stream stream = new FileStream($"{Global.Mark}\\{file}", FileMode.Create, FileAccess.Write, FileShare.None);
            formatter.Serialize(stream, _model);
            stream.Close();
        }

        private float Develop(float bestUas, bool write, DataSet set, string fileName, string name = "dev")
		{
			var sents = set.Sents;
			var start = DateTime.Now;
            var count = 0;
            var total = (float)sents.Count/100;
            foreach (var t in sents)
			{
				_model.PredictOne(t);
                Global.Logger.WriteConsole($"{++count / total:f2}\r");
            }
			var end = DateTime.Now;
			var result = set.Evaluate();
			Global.Logger.WriteLine(name+": "+result+", "+ $"time= {end - start:hh\\:mm\\:ss\\.fff}");
			var uas = result.UasNoPunc;
			if (write && uas > bestUas)
			{
				string outFile = $"{Global.Mark}\\{Path.GetFileNameWithoutExtension(fileName)}-dev-{uas:F}{Path.GetExtension(fileName)}";
				set.WriteFile(outFile);
			}
			return uas;
		}

        private void Develop(DataSet set, string fileName, string name = "dev")
        {
            var sents = set.Sents;
            var start = DateTime.Now;
            var count = 0;
            var total = (float)sents.Count / 100;
            foreach (var t in sents)
            {
                _model.TestOne(t);
                Global.Logger.WriteConsole($"{++count / total:f2}\r");
            }
            var end = DateTime.Now;
            var result = set.Evaluate();
            Global.Logger.WriteLine(name + ": " + result + ", " + $"time= {end - start:hh\\:mm\\:ss\\.fff}");
            var uas = result.UasNoPunc;
                string outFile = $"{Global.Mark}\\{Path.GetFileNameWithoutExtension(fileName)}-dev-{uas:F}{Path.GetExtension(fileName)}";
                set.WriteRecordFile(outFile);
            //return uas;
        }


        private void Test(float bestUas, DataSet set, string fileName, string name="tst")
		{
			var sents = set.Sents;
			var start = DateTime.Now;
            var count = 0;
            var total = (float)sents.Count / 100;
			foreach (var t in sents)
			{
				_model.PredictOne(t);
                Global.Logger.WriteConsole($"{++count / total:f2}\r");
			}
			var end = DateTime.Now;
            var result = set.Evaluate();
            Global.Logger.WriteLine(name + ": " + result + ", " + $"time= {end - start:hh\\:mm\\:ss\\.fff}");
            var span = end - start;
			var time = span.TotalMilliseconds;
			var nWords = sents.Sum(s => s.Count);
			Global.Logger.WriteLine(
				$"test: parsed {nWords} words in {sents.Count} sentences in {span} at {1000.0 * nWords / time:F} w/s, {1000.0 * sents.Count / time:F} sent/s.");
			var outFile =
				$"{Global.Mark}\\{Path.GetFileNameWithoutExtension(fileName)}-dev-{bestUas:F}{Path.GetExtension(fileName)}";
			set.WriteRecordFile(outFile);
		}



        public void Train()
		{
			Global.Logger.WriteLine($"{Config.Separator}");
			var bestUas = 0.0f;
			var report = (_train.Count + _conf.IterPerEpoch - 1) / _conf.IterPerEpoch;
			for (var epoch = 0; epoch < _conf.MaxEpoch; epoch++)
			{
				_train.Shuffle();
				for (var iter = 0; iter < _conf.IterPerEpoch; iter++)
				{
                    var start = DateTime.Now;
					var loss = 0d;
					var count = 0;
					for (var i = iter * report; count < report && i < _train.Count; i++,count++)
					{
						loss += _model.TrainOne(_train.ShuffledSents[i]);
                        Global.Logger.WriteConsole($"{(float)(count+1)*100/ report:f2}\r");
					}
					var end = DateTime.Now;
					Global.Logger.WriteLine($"{epoch}-{iter}: loss= {loss / count:f4}, time= {end - start:hh\\:mm\\:ss\\.fff}");
					var uas = Develop(bestUas, true, _dev, _conf.DevFile);

					if (uas > bestUas)
					{
						var testuas = -1.0;
						bestUas = uas;
						if (_test != null)
						{
							testuas = Develop(bestUas, true, _test, _conf.TestFile, "tst");
						}
						Global.Logger.WriteLine($"$ {uas:f2}{(testuas<0?"":$" -> {testuas:f2}")}");
                        Save($"dev.{uas:f2}.bin");
					}

                }
			}
			Global.Logger.WriteLine($"{Config.Separator}");
		}

        public void Dev()
        {
            Global.Logger.WriteLine($"{Config.Separator}");

            Develop(_dev, _conf.DevFile);
            if (_test != null)
            {
                Develop(_test, _conf.TestFile, "tst");
            }

            Global.Logger.WriteLine($"{Config.Separator}");
        }

        public void Tst()
        {
            Global.Logger.WriteLine($"{Config.Separator}");

            if (_dev != null)
            {
                Test(0, _dev, _conf.DevFile, "dev");
            }
            if (_test != null)
            {
                Test(0, _test, _conf.TestFile);
            }
            Global.Logger.WriteLine($"{Config.Separator}");
        }

	}
}