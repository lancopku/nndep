using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using nndep.Networks.Graph;

namespace nndep.Networks.Opts
{
	class Sgd : OptBase
	{
		private readonly float _decayRate;
		private readonly int _setSize;
		private int _pastCnt;


		public Sgd(float lr, float l2RegFactor, float clipRange, float dr, int setSize)
			: base(OptType.sgd, lr, clipRange, l2RegFactor)
		{
			_decayRate = dr;
			_setSize = setSize;
			_pastCnt = 0;
		}

		public override void Prepare(IEnumerable<Tensor> p)
		{
			// no extra storage needed
		}

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append(base.ToString());
			sb.AppendLine($"DecayRate = {_decayRate:g2}");
			return sb.ToString();
		}

		public override void Update(IEnumerable<Tensor> parameters, float scale)
		{
			Interlocked.Increment(ref _pastCnt);
			var newlr = LearningRate * (float)Math.Pow(_decayRate, (float) _pastCnt / _setSize);
			foreach (var p in parameters)
			{
				var w = p.W.Storage;
				var d = p.Grad.Storage;
				for (var i = 0; i < w.Length; i++)
				{
					var dw = d[i] * scale;
					if (dw > ClipRange)
					{
						dw = ClipRange;
					}
					else if (dw < -ClipRange)
					{
						dw = -ClipRange;
					}
					w[i] -= dw * newlr;
					d[i] = 0;
				}
			}
		}

        public override void Update(IEnumerable<Tensor> parameters)
        {
            throw new NotImplementedException();
        }
    }
}