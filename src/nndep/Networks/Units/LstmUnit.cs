using System;
using nndep.Networks.Graph;

namespace nndep.Networks.Units
{
    [Serializable]
    class LstmUnit
	{
		private readonly Tensor _bc;
		private readonly Tensor _bf;
		private readonly Tensor _bi;
		private readonly Tensor _bo;
		private readonly Tensor _wch;

		private readonly Tensor _wcx;
		private readonly Tensor _wfh;

		private readonly Tensor _wfx;
		private readonly Tensor _wih;
		private readonly Tensor _wix;
		private readonly Tensor _woh;

		private readonly Tensor _wox;

		public LstmUnit(int nIn, int nHid)
		{
			_wix = new Tensor(nIn, nHid, true);
			_wih = new Tensor(nHid, nHid, true);
			_bi = new Tensor(1, nHid, true);
			_wfx = new Tensor(nIn, nHid, true);
			_wfh = new Tensor(nHid, nHid, true);
			_bf = new Tensor(1, nHid, true);
			_wox = new Tensor(nIn, nHid, true);
			_woh = new Tensor(nHid, nHid, true);
			_bo = new Tensor(1, nHid, true);
			_wcx = new Tensor(nIn, nHid, true);
			_wch = new Tensor(nHid, nHid, true);
			_bc = new Tensor(1, nHid, true);
		}

		public void Init(Func<int, int, Func<float>> fac)
		{
			_wfx.Fill(fac(_wfx.Row, _wfx.Col));
			_wfh.Fill(fac(_wfh.Row, _wfh.Col));
			_wix.Fill(fac(_wix.Row, _wix.Col));
			_wih.Fill(fac(_wih.Row, _wih.Col));
			_wcx.Fill(fac(_wcx.Row, _wcx.Col));
			_wch.Fill(fac(_wch.Row, _wch.Col));
			_wox.Fill(fac(_wox.Row, _wox.Col));
			_woh.Fill(fac(_woh.Row, _woh.Col));
			_bf.Fill(() => 1);
		}

		// i(t) = sigmoid(W(i)*x(t) + U(i)*h(t-1))             input gate
		// f(t) = sigmoid(W(f)*x(t) + U(f)*h(t-1))             forget gate
		// o(t) = sigmoid(W(o)*x(t) + U(o)*h(t-1))             output exposure gate
		// c tilde(t) = tanh(W(c)*x(t) + U(c)*h(t-1))          new memory cell
		// c(t) = f(t).*c tilde(t-1) + i(t).*c tilde(t)        final memory cell
		// h(t) = o(t).*tanh(c(t))
		public Tensor Step(Graph.Graph f, Tensor x, Tensor c, Tensor h, out Tensor nc)
		{
			// input gate
			var inputGate = f.Sigmoid(f.AddBias(f.Add(f.Multiply(x, _wix), f.Multiply(h, _wih)), _bi));
			// forget gate
			var forgetGate = f.Sigmoid(f.AddBias(f.Add(f.Multiply(x, _wfx), f.Multiply(h, _wfh)), _bf));
			// output gate
			var outputGate = f.Sigmoid(f.AddBias(f.Add(f.Multiply(x, _wox), f.Multiply(h, _woh)), _bo));
			// new cell
			var newInfo = f.Tanh(f.AddBias(f.Add(f.Multiply(x, _wcx), f.Multiply(h, _wch)), _bc));
			// final cell
			var retainCell = f.ElementwiseMultiply(forgetGate, c);
			var writeCell = f.ElementwiseMultiply(inputGate, newInfo);
			var finalCell = f.Add(retainCell, writeCell);
			// hidden unit
			nc = finalCell;
			return f.ElementwiseMultiply(outputGate, f.Tanh(finalCell));
		}


		public Tensor[] Step(Graph.Graph f, Tensor[] x, bool reverse = false)
		{
			var res = new Tensor[x.Length];
			var cells = new Tensor[x.Length];
			var c = new Tensor(1, _bc.Col, true);
			var h = new Tensor(1, _bo.Col, true);
			if (reverse)
			{
				for (var i = x.Length - 1; i >= 0; i--)
				{
					res[i] = Step(f, x[i], c, h, out cells[i]);
					h = res[i];
					c = cells[i];
				}
			}
			else
			{
				for (var i = 0; i < x.Length; i++)
				{
					res[i] = Step(f, x[i], c, h, out cells[i]);
					h = res[i];
					c = cells[i];
				}
			}
			return res;
		}

		public Tensor[] SubmitParameters()
		{
			return new[]
			{
				_wix,
				_wih,
				_bi,
				_wfx,
				_wfh,
				_bf,
				_wox,
				_woh,
				_bo,
				_wcx,
				_wch,
				_bc
			};
		}
	}
}