using System;
using nndep.Networks.Graph;

namespace nndep.Networks.Units
{
    [Serializable]
    class DenseUnit
	{
		private readonly Tensor _bh;
		private readonly Tensor _wxh;

		public DenseUnit(int nIn, int nHid)
		{
			_wxh = new Tensor(nIn, nHid, true);
			_bh = new Tensor(1, nHid, true);
		}

		public void Init(Func<int, int, Func<float>> fac)
		{
			_wxh.Fill(fac(_wxh.Row, _wxh.Col));
		}

        public void Init(DenseUnit d)
        {
            _wxh.Fill(d._wxh.W.Storage);
            _bh.Fill(d._bh.W.Storage);
        }


		public Tensor Step(Graph.Graph f, Tensor x)
		{
			return f.AddBias(f.Multiply(x, _wxh), _bh);
		}

		public Tensor[] SubmitParameters()
		{
			return new[] {_wxh, _bh};
		}
	}
}