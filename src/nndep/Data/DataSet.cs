using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using nndep.Util;

namespace nndep.Data
{
	//using Sent = Sent;


	class DataSet
	{
		public Map<string> DepLabel;
		public Map<string> Form;
		public Map<string> PosTag;
		public List<Sent> Sents;
        public List<Sent> NonProjectvie;
        public string RootLabel;

		public List<Sent> ShuffledSents;
	    private readonly RandomNumberGenerator _rng;

		public DataSet(string inFile, bool useCpos, RandomNumberGenerator rng=null)
		{
			Form = null;
			PosTag = null;
			DepLabel = null;
			ShuffledSents = null;
            RootLabel = null;
			Sents = LoadFile(inFile, useCpos).ToList();
            NonProjectvie = null;
		    _rng = rng;
		}

		public int Count => Sents.Count;


        public static Result Evaluate(Sent sent)
        {
            var puncs = GetPunctuationTags();
            var correctHeads = 0;
            var correctHeadsNoPunc = 0;
            var sumArcs = 0;
            var sumArcsNoPunc = 0;
            var correctArcsNoPunc = 0;
            var correctArcs = 0;

            foreach (var ent in sent)
            {
                if (ent.PredParId == ent.Pid)
                {
                    ++correctHeads;
                    if (ent.PredRelation == ent.Relation)
                    {
                        ++correctArcs;
                    }
                }
                ++sumArcs;

                var tag = ent.Pos;
                if (!puncs.Contains(tag) && ent.Pos != "PUNCT" && ent.Pos != "PU")
                {
                    ++sumArcsNoPunc;
                    if (ent.PredParId == ent.Pid)
                    {
                        ++correctHeadsNoPunc;
                        if (ent.PredRelation == ent.Relation)
                        {
                            ++correctArcsNoPunc;
                        }
                    }
                }
            }

            var correctTree = correctHeads == sent.Count ? 1 : 0;
            var correctTreeNoPunc = correctHeadsNoPunc == sumArcsNoPunc ? 1 : 0;
            var correctRoot = sent.GetPRoot() == sent.GetRoot() ? 1 : 0;

            return new Result
            {
                Uas = (float)correctHeads  / sumArcs,
                UasNoPunc = (float)correctHeadsNoPunc  / sumArcsNoPunc,
                Las = (float)correctArcs  / sumArcs,
                LasNoPunc = (float)correctArcsNoPunc  / sumArcsNoPunc,
                Uem = correctTree,
                UemNoPunc = correctTreeNoPunc,
                Root = correctRoot
            };

        }


		public Result Evaluate()
		{
			var punctuationTags = GetPunctuationTags();

			var correctArcs = 0;
			var correctArcsNoPunc = 0;
			var correctHeads = 0;
			var correctHeadsNoPunc = 0;

			var correctTrees = 0;
			var correctTreesNoPunc = 0;
			var correctRoot = 0;

			var sumArcs = 0;
			var sumArcsNoPunc = 0;

			foreach (var sent in Sents)
			{
				var nCorrectHead = 0;
				var nCorrectHeadNoPunc = 0;
				var nNoPunc = 0;

				foreach (var ent in sent)
				{
					if (ent.PredParId == ent.Pid)
					{
						++correctHeads;
						++nCorrectHead;
						if (ent.PredRelation == ent.Relation)
						{
							++correctArcs;
						}
					}
					++sumArcs;

					var tag = ent.Pos;
					if (!punctuationTags.Contains(tag) && ent.Pos!="PUNCT" && ent.Pos!="PU")
					{
						++sumArcsNoPunc;
						++nNoPunc;
						if (ent.PredParId == ent.Pid)
						{
							++correctHeadsNoPunc;
							++nCorrectHeadNoPunc;
							if (ent.PredRelation == ent.Relation)
							{
								++correctArcsNoPunc;
							}
						}
					}
				}
				// sent does not have root entry
				if (nCorrectHead == sent.Count)
					++correctTrees;
				if (nCorrectHeadNoPunc == nNoPunc)
					++correctTreesNoPunc;
				if (sent.GetPRoot() == sent.GetRoot())
					++correctRoot;
			}

			return new Result
			{
				Uas = correctHeads * 100.0f / sumArcs,
				UasNoPunc = correctHeadsNoPunc * 100.0f / sumArcsNoPunc,
				Las = correctArcs * 100.0f / sumArcs,
				LasNoPunc = correctArcsNoPunc * 100.0f / sumArcsNoPunc,
				Uem = correctTrees * 100.0f / Sents.Count,
				UemNoPunc = correctTreesNoPunc * 100.0f / Sents.Count,
				Root = correctRoot * 100.0f / Sents.Count
			};
		}


		public void GenerateMappings(int formCutOff)
		{
			var word = new List<string>();
			var pos = new List<string>();
			var label = new List<string>();

			foreach (var sentence in Sents)
			{
				foreach (var entry in sentence)
				{
					word.Add(entry.Norm);
					pos.Add(entry.Pos);
				}
			}

			string rootLabel = null;


			foreach (var sentence in Sents)
			{
				foreach (var entry in sentence)
				{
					if (entry.Pid == 0)
					{
						rootLabel = entry.Relation;
					}
					else
					{
						label.Add(entry.Relation);
					}
				}
			}

			Form = new Map<string>(word, new[] {Config.Root}, formCutOff);
			PosTag = new Map<string>(pos, new[] {Config.Root});
			DepLabel = new Map<string>(label, new[] {rootLabel});
            RootLabel = rootLabel;
			Global.Logger.WriteLine(Config.Separator);
			Global.Logger.WriteLine("form = " + Form.Count);
			Global.Logger.WriteLine("postag = " + PosTag.Count);
			Global.Logger.WriteLine("dep = " + DepLabel.Count);
		}

		public static HashSet<string> GetPunctuationTags()
		{

			return new HashSet<string>(new[] {",", ".", "``", "''", ":"});
		}

		private IEnumerable<Sent> LoadFile(string inFile, bool useCpos)
		{
			using (var reader = new StreamReader(inFile, Encoding.UTF8))
			{
				var sent = new Sent();
				var line = reader.ReadLine();
				while (line != null)
				{
					if (line.StartsWith("#"))
					{
						line = reader.ReadLine();
						continue;
					}
					if (string.IsNullOrWhiteSpace(line))
					{
						yield return sent;
						sent = new Sent();
					}
					else
					{
						// line is ID Form Lemma CPOS POS Feature PID Relation _ _
						var splits = line.Split();
                        var single = int.TryParse(splits[0], out _);
                        if (single)
                        {
                            sent.Add(new Entry(int.Parse(splits[0]), splits[1], useCpos?splits[3]:splits[4], int.Parse(splits[6]), splits[7]));
                        }
					}
					line = reader.ReadLine();
				}
				if (sent.Count != 0)
				{
					yield return sent;
				}
			}
		}

		public void PrintTreeStats(bool remove=true)
		{
			Global.Logger.WriteLine(Config.Separator);
			var nTrees = Sents.Count;
			var nonTree = 0;
			var multiRoot = 0;
			var nonProjective = 0;
			foreach (var sent in Sents.ToArray())
			{
				if (!sent.IsTree())
				{
                    if(remove)
                        Sents.Remove(sent);
					++nonTree;
				}
				else
				{
					if (!sent.IsProj())
					{
                        if (remove)
                            Sents.Remove(sent);
                        if(NonProjectvie == null)
                        {
                            NonProjectvie = new List<Sent>();
                        }
                        NonProjectvie.Add(sent);
						++nonProjective;
					}
					if (!sent.IsSingleRoot())
					{
                        if (remove)
                            Sents.Remove(sent);
						++multiRoot;
					}
				}
			}
			Global.Logger.WriteLine($"sentences = {nTrees}");
			Global.Logger.WriteLine($"illegal = {nonTree} ({(float)nonTree/ nTrees:p})");
			Global.Logger.WriteLine($"multiple roots = {multiRoot} ({(float)multiRoot/ nTrees:p})");
			Global.Logger.WriteLine($"not projective = {nonProjective} ({(float)nonProjective/ nTrees:p})");
		}

		public void Shuffle()
		{
			if (ShuffledSents == null)
			{
				ShuffledSents = new List<Sent>(Sents);
			}
			Sents.Shuffle(_rng);
		}

		public void WriteFile(string outFile)
		{
			using (var fout = new StreamWriter(outFile, false, Encoding.UTF8))
			{
				foreach (var sent in Sents)
				{
					foreach (var entry in sent)
					{
						if (entry.Id == 0)
						{
							continue;
						}
						// line is ID Form Lemma CPOS POS Feature PID Relation _ _
						fout.WriteLine($"{entry.Id}\t{entry.Form}\t_\t_\t_\t_\t{entry.PredParId}\t{entry.PredRelation}\t_\t_");
					}
					fout.WriteLine();
				}
			}
		}

        public void WriteRecordFile(string outFile)
        {
            using (var fout = new StreamWriter(outFile, false, Encoding.UTF8))
            {
                foreach (var sent in Sents)
                {
                    foreach (var entry in sent)
                    {
                        if (entry.Id == 0)
                        {
                            continue;
                        }
                        // line is ID Form Lemma CPOS POS Feature PID Relation _ _
                        fout.WriteLine($"{entry.Id}\t{entry.Form}\t_\t_\t_\t_\t{entry.PredParId}\t{entry.PredRelation}\t_\t_");
                    }
                    //fout.WriteLine("# Gold seq");
                    //foreach(var record in sent.trainRecords)
                    //{
                    //    fout.WriteLine("# "+record.ToString());
                    //}
                    //fout.WriteLine("# Pred seq");
                    //foreach (var record in sent.predRecords)
                    //{
                    //    fout.WriteLine("# " + record.ToString());
                    //}
                    //fout.WriteLine();
                }
            }
        }


        public class Result
		{
			public float Uas { get; set; }
			public float UasNoPunc { get; set; }
			public float Las { get; set; }
			public float LasNoPunc { get; set; }
			public float Uem { get; set; }

			public float UemNoPunc { get; set; }

			public float Root { get; set; }

			public override string ToString()
			{
				var sb = new StringBuilder();
				sb.Append($"uas = {Uas:f2}/{UasNoPunc:f2}, ");
				sb.Append($"las = {Las:f2}/{LasNoPunc:f2}, ");
				sb.Append($"uem = {Uem:f2}/{UemNoPunc:f2}, ");
				sb.Append($"root = {Root:f2}");
				return sb.ToString();
			}
		}
	}
}