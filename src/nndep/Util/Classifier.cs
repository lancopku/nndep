using System;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using DepParser.Networks;
using DepParser.Networks.Opts;
using DepParser.Networks.Units;

namespace DepParser.Common
{

    class Classifier
    {
        private double[][] _embedd;
        //public NetBase Net;
        public OptBase Opt;

        private readonly DataSet _dataset;
        private readonly int _labelCount;


        public Classifier(DataSet dataset, int transitionCount)
        {
            Global.Logger.WriteLine($"{Config.Separator}{Config.Separator} Init Start");
            _dataset = dataset;
            _labelCount = transitionCount;            
            InitEmbed(Global.Config.EmbedFile);
            InitNet(Global.Config.NetType, Global.Config.OptType);
            Global.Logger.WriteLine($"{Config.Separator}{Config.Separator} Init End");

        }


        private void InitEmbed(string embedFile)
        {
            Global.Logger.WriteLine(Config.Separator + "Init Embed");
            _embedd = new double[_dataset.TotalCount][];
            Dictionary<string, double[]> embeds = null;
            if (embedFile != null && File.Exists(embedFile))
            {
                embeds = FileUtil.ReadEmbedFile(embedFile);
            }
            var foundEmbed = 0;
            var allWords = _dataset.Words;
            foreach (var word in allWords)
            {
                if ((embeds?.ContainsKey(word) ?? false )|| (embeds?.ContainsKey(word.ToLower()) ?? false))
                {
                    var wordId = _dataset.GetKnownWordId(word);
                    _embedd[wordId] = embeds.ContainsKey(word)
                        ? embeds[word]
                        : (embeds[word.ToLower()].Clone() as double []);
                    foundEmbed++;
                }
            }
            for (var i = 0; i < _embedd.Length; i++)
            {
                if (_embedd[i] == null)
                {
                    _embedd[i] = new double[Global.Config.EmbeddingSize];
                    for (var j = 0; j < _embedd[i].Length; j++)
                    {
                        _embedd[i][j] = RandomNumberGenerator.GetDouble(-Global.Config.InitRange, Global.Config.InitRange);
                    }
                }
            }

            Global.Logger.WriteLine($"Found Embeddings: {foundEmbed}/{allWords.Count}");
        }

        private void InitNet(NetType netType, OptType optType)
        {
            switch (optType)
            {
                case OptType.sgd:
                    Opt = new Sgd(Global.Config.LearningRate, Global.Config.L2RegFactor, Global.Config.ClipBound, Global.Config.DecayRate,
                        _dataset.Examples.Count);
                    break;
                case OptType.adagrad:
                    Opt = new AdaGrad(Global.Config.LearningRate, Global.Config.L2RegFactor, Global.Config.ClipBound, Global.Config.Eps);
                    break;
                case OptType.adam:
                    Opt = new Adam(Global.Config.LearningRate, Global.Config.L2RegFactor, Global.Config.ClipBound, 0.999f, 0.9f, 1e-8f);
                    break; ;
                default:
                    throw new ArgumentOutOfRangeException(nameof(optType), optType, null);
            }

            var eDim = Global.Config.TokenCount * Global.Config.EmbeddingSize;
            var hDim = Global.Config.HiddenSize;
            var oDim = _labelCount;

            switch (netType)
            {
                case NetType.ffnn:
                    Net = new FFNN(eDim, hDim, oDim, Opt);
                    break;
                case NetType.blstm:
                    Net = new Blstm(eDim, hDim, oDim, Opt);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(netType), netType, null);
            }

            Net.InitEmbed(_embedd);
        }


        public void TrainOnline(Example ex)
        {
            var feature = ex.Feature;
            var label = ex.Label;
            Net.TrainOne(feature, label);
        }


        public void TrainMinibatch(List<Example> examples)
        {
            if (Global.Config.TrainingThreads == 1)
            {
                for(var i=0;i< examples.Count;i++)
                {
                    var ex = examples[i];
                    Net.TrainOneWithoutUpdate(ex.Feature, ex.Label);
                }
            }
            else
            {
                Parallel.For(0, examples.Count, new ParallelOptions() { MaxDegreeOfParallelism = Global.Config.TrainingThreads }, i =>
                {
                    var ex = examples[i];
                    Net.TrainOneWithoutUpdate(ex.Feature, ex.Label);
                });
            }
            Net.Update(1.0/examples.Count);
        }

        public double[] Predict(int[] feature)
        {
            return Net.PredictOne(feature);
        }
    }
}
