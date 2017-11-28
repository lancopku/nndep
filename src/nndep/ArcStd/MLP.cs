using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using nndep.Data;
using nndep.Networks;
using nndep.Networks.Graph;
using nndep.Networks.Inits;
using nndep.Networks.Opts;
using nndep.Networks.Units;
using nndep.Util;

namespace nndep.ArcStd
{
    [Serializable]
    class MLP : Model
    {
        // model from 
        //     Incremental Parsing with Minimal Features Using Bi-Directional LSTM
        //     J. Cross & L. Huang
        //     https://arxiv.org/pdf/1606.06406.pdf
        //
        //       actOutput/labelOutput       3/nLabel
        //              |relu
        //           act/label                nHid
        //    |        |        |
        //  stack(1) stack(0) buffer(0)      3*2*nHid
        //
        //          repr
        //           |dropout
        //         blstm                         2*nHid
        //         |    |
        //        pos word                      2*nEmb




        [NonSerialized] private readonly RandomNumberGenerator _rng;
        [NonSerialized] private readonly RandomNumberGenerator _rngChoice;
        [NonSerialized] private Config _conf;
        [NonSerialized] private DataSet _train;
        [NonSerialized] private OptBase _opt;

        private readonly Tensor _non;
        private readonly BiLstmUnit _surface;
        private readonly Tensor[] _formemb;
        private readonly Tensor[] _posemb;

        private readonly DenseUnit _act;
        private readonly DenseUnit _actOutput;
        private readonly DenseUnit _label;
        private readonly DenseUnit _labelOutput;
        

        public MLP(Config conf, DataSet train)
        {
            _train = train;
            _conf = conf;
            _rng = new RandomNumberGenerator(conf.Seed);
            _rngChoice = new RandomNumberGenerator(conf.Seed);
            _formemb = File.Exists(conf.EmbedFile)
                ? InitEmbed(conf.EmbedFile, train.Form, conf.InitRange)
                : InitEmbed(train.Form, conf.InitRange);
            _posemb = InitEmbed(train.PosTag, conf.InitRange);
            _non = new Tensor(1, conf.HiddenSize * 2, true);
            _surface = new BiLstmUnit(conf.EmbeddingSize * 2, conf.HiddenSize);
            _act = new DenseUnit(conf.HiddenSize * 6, conf.HiddenSize);
            _actOutput = new DenseUnit(conf.HiddenSize, 3);
            _label = new DenseUnit(conf.HiddenSize * 6, conf.HiddenSize);
            _labelOutput = new DenseUnit(conf.HiddenSize * 3, train.DepLabel.Count);
            FixedParams.AddRange(_surface.SubmitParameters());
            FixedParams.AddRange(_act.SubmitParameters());
            FixedParams.AddRange(_label.SubmitParameters());
            FixedParams.AddRange(_actOutput.SubmitParameters());
            FixedParams.AddRange(_labelOutput.SubmitParameters());
            FixedParams.Add(_non);
            AllParams.AddRange(FixedParams);
            AllParams.AddRange(_formemb);
            AllParams.AddRange(_posemb);


            _surface.Init((fin, fout) => new GlorotNormalInit(fin, fout, _rng).Next);
            _act.Init((fin, fout) => new GlorotNormalInit(fin, fout, _rng).Next);
            _label.Init((fin, fout) => new GlorotNormalInit(fin, fout, _rng).Next);
            _actOutput.Init((fin, fout) => new GlorotNormalInit(fin, fout, _rng).Next);
            _labelOutput.Init((fin, fout) => new GlorotNormalInit(fin, fout, _rng).Next);

            switch (conf.OptType)
            {
                case OptType.sgd:
                    _opt = new Sgd(conf.LearningRate, conf.L2RegFactor, conf.ClipBound, conf.DecayRate,
                        train.Count);
                    break;
                case OptType.adagrad:
                    _opt = new AdaGrad(conf.LearningRate, conf.L2RegFactor, conf.ClipBound, conf.Eps);
                    break;
                case OptType.adam:
                    _opt = new Adam(conf.LearningRate, conf.L2RegFactor, conf.ClipBound, 0.999f, 0.9f, 1e-8f);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(conf.OptType), "unknown opt type");
            }
            _opt.Prepare(AllParams);

            G.SetRng(conf.Seed);
        }


        public void Recover(Config conf, DataSet train)
        {
            _train = train;
            _conf = conf;
            FixedParams.AddRange(_surface.SubmitParameters());
            FixedParams.AddRange(_act.SubmitParameters());
            FixedParams.AddRange(_label.SubmitParameters());
            FixedParams.AddRange(_actOutput.SubmitParameters());
            FixedParams.AddRange(_labelOutput.SubmitParameters());
            FixedParams.Add(_non);
            AllParams.AddRange(FixedParams);
            AllParams.AddRange(_formemb);
            AllParams.AddRange(_posemb);
            switch (_conf.OptType)
            {
                case OptType.sgd:
                    _opt = new Sgd(_conf.LearningRate, _conf.L2RegFactor, _conf.ClipBound, _conf.DecayRate,
                        train.Count);
                    break;
                case OptType.adagrad:
                    _opt = new AdaGrad(_conf.LearningRate, _conf.L2RegFactor, _conf.ClipBound, _conf.Eps);
                    break;
                case OptType.adam:
                    _opt = new Adam(_conf.LearningRate, _conf.L2RegFactor, _conf.ClipBound, 0.999f, 0.9f, 1e-8f);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(_conf.OptType), "unknown opt type");
            }
        }


        private void GetTokenRepr(Sent sent, bool grad)
        {
            var x = new Tensor[sent.Count];
            for (var i = 0; i < sent.Count; i++)
            {
                var form = _formemb[_train.Form[sent[i].Norm]];
                form.RefCount++;
                var pos = _posemb[_train.PosTag[sent[i].Pos]];
                pos.RefCount++;
                if (grad)
                {
                    VariedParams.Add(form);
                    VariedParams.Add(pos);
                }
                x[i] = G.Concat(form, pos);
            }
            var t = _surface.Step(G, x);
            for (var i = 0; i < sent.Count; i++)
            {
                sent[i].Repr = G.DropOut(t[i], 0.5f);
            }
        }

        private void GetTrans(ArcStandardConfig c, out Tensor op, out Tensor label)
        {
            var left = c.Stack.Back(1);
            var right = c.Stack.Back(0);
            var next = c.Buffer.Front(0);
            var comb = G.Concat(left?.Repr ?? _non, right?.Repr ?? _non, next?.Repr ?? _non);
            _non.RefCount += left == null ? 1 : 0;
            _non.RefCount += right == null ? 1 : 0;
            _non.RefCount += next == null ? 1 : 0;
            op = _actOutput.Step(G, G.Relu(_act.Step(G, comb)));
            label = _labelOutput.Step(G, G.Relu(_label.Step(G, comb)));
        }

        private Tensor[] InitEmbed(string embedFile, Map<string> form, float range)
        {
            var size = _conf.EmbeddingSize;
            //Global.Logger.WriteLine(Config.Separator + "Init Embed");
            var extemb = ReadEmbedFile(embedFile);
            var nEmb = form.MaxIdx + 1;
            var emb = new Tensor[nEmb];
            var foundEmbed = 0;
            for (var i = 0; i < nEmb; i++)
            {
                emb[i] = new Tensor(1, size, true);
                var f = form[i];
                if (f != null)
                {
                    if (extemb.ContainsKey(f))
                    {
                        emb[i].Fill(extemb[f]);
                        foundEmbed++;
                    }
                    else if (extemb.ContainsKey(f.ToLower()))
                    {
                        emb[i].Fill(extemb[f.ToLower()]);
                        foundEmbed++;
                    }
                }
                emb[i].Fill(() => _rng.GetFloat(-range, range));
            }
            Global.Logger.WriteLine($"Found Embeddings: {foundEmbed}/{nEmb}");
            return emb;
        }

        private Tensor[] InitEmbed(Map<string> map, float range)
        {
            var size = _conf.EmbeddingSize;
            var nEmb = map.MaxIdx + 1;
            var emb = new Tensor[nEmb];
            for (var i = 0; i < nEmb; i++)
            {
                emb[i] = new Tensor(1, size, true);
                emb[i].Fill(() => _rng.GetFloat(-range, range));
            }
            return emb;
        }


        public void TestOne(Sent sent)
        {

            G.Need = false;

            var c = new ArcStandardConfig(sent, false, null);
            GetTokenRepr(sent, false);
            while (!c.IsTerminal())
            {
                GetTrans(c, out Tensor op, out Tensor label);

                // in labelid 0 is root
                // in mapping 1 is root, 0 is always unk
                var labelid = label.W.MaxIndex();
                var plabel = _train.DepLabel[labelid + 1];


                var opid = op.W.MaxIndex();
                var optTrans = (ArcStandardConfig.Op) opid;
                c.Apply(optTrans, plabel);
            }


        }


        public void PredictOne(Sent sent)
        {
            G.Need = false;

            var c = new ArcStandardConfig(sent, false, null);
            GetTokenRepr(sent, false);
            while (!c.IsTerminal())
            {
                GetTrans(c, out Tensor op, out Tensor label);

                // in labelid 0 is root
                // in mapping 1 is root, 0 is always unk
                var labelid = label.W.MaxIndex();
                var plabel = _train.DepLabel[labelid + 1];
                var scores = op.W.Storage;
                var optScore = float.NegativeInfinity;
                var optTrans = ArcStandardConfig.Op.shift;

                for (var j = 0; j < 3; ++j)
                {
                    if (scores[j] > optScore && c.CanApply((ArcStandardConfig.Op) j, plabel))
                    {
                        optScore = scores[j];
                        optTrans = (ArcStandardConfig.Op) j;
                    }
                }
                c.Apply(optTrans, plabel);
            }
        }


        private Dictionary<string, float[]> ReadEmbedFile(string embedFile)
        {
            var embeddings = new Dictionary<string, float[]>();
            using (var reader = new StreamReader(embedFile, Encoding.UTF8))
            {
                var line = reader.ReadLine();
                while (line != null)
                {
                    var splits = line.Split();
                    var n = splits.Length - 1;
                    var vals = new float[n];
                    for (var i = 0; i < n; i++)
                    {
                        vals[i] = float.Parse(splits[i + 1]);
                    }
                    embeddings[splits[0].Trim()] = vals;
                    line = reader.ReadLine();
                }
            }
            return embeddings;
        }

        public float TrainOne(Sent sent)
        {
            G.Need = true;
            var c = new ArcStandardConfig(sent, false, _rngChoice);
            GetTokenRepr(sent, true);
            var losses = 0f;
            while (!c.IsTerminal())
            {
                var oracle = c.GetOracle(_conf.OraType, out float[] target, out string slabel);
                GetTrans(c, out Tensor op, out Tensor label);
                G.SoftmaxWithCrossEntropy(op, target, out float loss);
                losses += loss;
                if (slabel != null)
                {
                    // in mapping 0 is unk, 1 is root
                    var labelid = _train.DepLabel[slabel] - 1;
                    G.SoftmaxWithCrossEntropy(label, labelid, out loss);
                    losses += loss;
                }

                c.Apply(oracle, slabel);
            }
            G.Backward();
            _opt.Update(FixedParams);
            _opt.Update(VariedParams);
            VariedParams.Clear();
            G.Clear();
            return losses / sent.Count;
        }
    }
}