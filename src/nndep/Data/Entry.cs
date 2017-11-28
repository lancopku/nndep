using System.Text.RegularExpressions;
using nndep.Networks.Graph;
using System.Collections.Generic;
using System.Linq;

namespace nndep.Data
{




    class Sent: List<Entry>
    {

        public int GetPRoot()
        {
            foreach (var ent in this)
            {
                if (ent.PredParId == 0)
                {
                    return ent.Id;
                }
            }
            return -1;
        }


        public int GetRoot()
        {
            foreach (var ent in this)
            {
                if (ent.Pid == 0)
                {
                    return ent.Id;
                }
            }
            return -1;
        }

        public bool IsProj()
        {
            var counter = -1;
            return VisitTree(this, 0, ref counter);
        }

        public bool IsSingleRoot()
        {
            var root = this.Count(x => x.Pid == 0);
            return root == 1;
        }

        public bool IsTree()
        {
            var nent = Count;
            var h = new List<int> { -1 };
            for (var i = 0; i < nent; ++i)
            {
                var hh = this[i].Id;
                if (hh < 0 || hh > nent)
                {
                    return false;
                }
                h.Add(-1);
            }
            for (var i = 1; i <= nent; ++i)
            {
                var dep = i;
                while (dep > 0)
                {
                    if (h[dep] >= 0 && h[dep] < i)
                    {
                        break;
                    }
                    if (h[dep] == i)
                    {
                        return false;
                    }
                    h[dep] = i;
                    dep = this[dep - 1].Pid;
                }
            }
            return true;
        }


        private bool VisitTree(Sent sent, int rootid, ref int counter)
        {
            // id start from 1
            // i start from 0
            // that is sent[i].id = i+1
            var rootidx = rootid - 1;
            for (var i = 0; i < rootidx; ++i)
            {
                if (sent[i].Pid == rootid && VisitTree(sent, sent[i].Id, ref counter) == false)
                {
                    return false;
                }
            }
            counter = counter + 1;
            if (rootid != counter)
            {
                return false;
            }
            for (var i = rootidx + 1; i < sent.Count; ++i)
            {
                if (sent[i].Pid == rootid && VisitTree(sent, sent[i].Id, ref counter) == false)
                {
                    return false;
                }
            }
            return true;
        }

    }


	class Entry
	{
		private static Regex _rgx = new Regex("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+");
		public readonly string Form;
		public readonly int Id;
		public readonly string Norm;
		public readonly int Pid;
		public readonly string Pos;
		public readonly string Relation;


		public int PredParId;
		public string PredRelation;
		public Tensor Repr;


		public Entry(int id, string form, string pos, int pid, string rel)
		{
			Id = id;
			Form = form;
			Norm = Normalize(form);
			Pos = pos;
			Pid = pid;
			Relation = rel;
		}


		private string Normalize(string word)
		{
			if (_rgx.IsMatch(word))
			{
				return "/num";
			}
			return word;
		}

		public static Entry Root()
		{
			return new Entry(0, Config.Root, Config.Root, -1, null);
		}

		public void SetHead(Entry ent, string relation)
		{
			PredParId = ent.Id;
            PredRelation = relation;
		}

		public override string ToString()
		{
			return $"{Id}:{Norm}[{Pos}]->{Pid}[{Relation}]/{PredParId}[{PredRelation}]";
		}
	}
}