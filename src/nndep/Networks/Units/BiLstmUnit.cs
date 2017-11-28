using System;
using System.Linq;
using nndep.Networks.Graph;

namespace nndep.Networks.Units
{
    [Serializable]
    class BiLstmUnit
	{
		private readonly LstmUnit _b;
		private readonly LstmUnit _f;

		public BiLstmUnit(int nIn, int nHid)
		{
			_f = new LstmUnit(nIn, nHid);
			_b = new LstmUnit(nIn, nHid);
		}

		public void Init(Func<int, int, Func<float>> fac)
		{
			_f.Init(fac);
			_b.Init(fac);
		}

		public Tensor[] Step(Graph.Graph f, Tensor[] x)
		{
			var res = new Tensor[x.Length];

			var fhs = _f.Step(f, x, false);
			var bhs = _f.Step(f, x, true);

			for (var i = 0; i < x.Length; i++)
			{
				res[i] = f.Concat(fhs[i], bhs[i]);
			}
			return res;
		}

		public Tensor[] SubmitParameters()
		{
			return _f.SubmitParameters().Concat(_b.SubmitParameters()).ToArray();
		}
	}
}