using System;
using nndep.Networks.Graph;

namespace nndep.Networks.Units
{
    [Serializable]
    class RecurrentUnit
	{
		private readonly Tensor _bh;
		private readonly Tensor _whh;
		private readonly Tensor _wxh;

		public RecurrentUnit(int nIn, int nHid)
		{
			_wxh = new Tensor(nIn, nHid, true);
			_whh = new Tensor(nHid, nHid, true);
			_bh = new Tensor(1, nHid, true);
		}

		public void Init(Func<int, int, Func<float>> fac)
		{
			_wxh.Fill(fac(_wxh.Row, _wxh.Col));
			_whh.Fill(fac(_whh.Row, _whh.Col));
		}

		public Tensor Step(Graph.Graph f, Tensor x, Tensor h)
		{
			return f.AddBias(f.Add(f.Multiply(x, _wxh), f.Multiply(h, _whh)), _bh);
		}

		public Tensor[] Step(Graph.Graph f, Tensor[] x)
		{
			var res = new Tensor[x.Length];
			var h = new Tensor(1, _bh.Col, true);
			for (var i = 0; i < x.Length; i++)
			{
				res[i] = Step(f, x[i], h);
				h = res[i];
			}
			return res;
		}

		public Tensor[] SubmitParameters()
		{
			return new[] {_wxh, _whh, _bh};
		}
	}
}