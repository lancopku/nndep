using System;
using System.Collections.Generic;
using System.Text;
using System.Threading;
using nndep.Networks.Graph;

namespace nndep.Networks.Opts
{
	class Adam : OptBase
	{
		private readonly float _decayRate;
		private readonly float _decayRate2;
		private readonly float _eps;
		private readonly float _keepRate;
		private readonly float _keepRate2;
		private int _times;

		// lr = 0.001, dr = 0.999, dr2 = 0.9, eps = 1e-8
		public Adam(float lr, float l2RegFactor, float clipRange, float dr, float dr2, float eps)
			: base(OptType.adam, lr, clipRange, l2RegFactor)
		{
			_decayRate = dr;
			_keepRate = 1 - dr;
			_decayRate2 = dr2;
			_keepRate2 = 1 - dr2;
			_eps = eps;
			_times = 0;
		}

		public override void Prepare(IEnumerable<Tensor> p)
		{
			foreach (var t in p)
			{
				t.InitParam(true, true);
			}
		}

		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append(base.ToString());
			sb.AppendLine($"beta1 = {_decayRate2:g2}");
			sb.AppendLine($"beta2 = {_decayRate:g2}");
			sb.AppendLine($"Eps = {_eps:g2}");
			return sb.ToString();
		}


		public override void Update(IEnumerable<Tensor> parameters, float scale)
		{
			Interlocked.Increment(ref _times);
			var bc1 = 1 - (float)Math.Pow(_decayRate, _times);
			var bc2 = 1 - (float)Math.Pow(_decayRate2, _times);
			foreach (var p in parameters)
			{
				var w = p.W.Storage;
				var d = p.Grad.Storage;
				var h = p.VarMom.Storage;
				var h2 = p.Mom.Storage;
				for (var i = 0; i < w.Length; i++)
				{
                    if (d[i] == 0)
                    {
                        continue;
                    }
					var dw = d[i] * scale + L2RegFactor * w[i];
					if (dw > ClipRange)
					{
						dw = ClipRange;
					}
					else if (dw < -ClipRange)
					{
						dw = -ClipRange;
					}
					var hw = h[i] * _decayRate + dw * dw * _keepRate;
					h[i] = hw;
					var hw2 = h2[i] * _decayRate2 + dw * _keepRate2;
					h2[i] = hw2;
					w[i] -= LearningRate * (hw2 / bc2) / ((float)Math.Sqrt(hw / bc1) + _eps);
					d[i] = 0;
				}
                p.RefCount = 0;
			}
		}

        public override void Update(IEnumerable<Tensor> parameters)
        {
            Interlocked.Increment(ref _times);
            var bc1 = 1 - (float)Math.Pow(_decayRate, _times);
            var bc2 = 1 - (float)Math.Pow(_decayRate2, _times);
            foreach (var p in parameters)
            {
                var scale = 1f;
                if(p.RefCount == 0)
                {
                    continue;
                }
                else
                {
                    scale = 1.0f / p.RefCount;
                    p.RefCount = 0;
                }
                var w = p.W.Storage;
                var d = p.Grad.Storage;
                var h = p.VarMom.Storage;
                var h2 = p.Mom.Storage;
                
                for (var i = 0; i < w.Length; i++)
                {
                    if (d[i] == 0)
                    {
                        continue;
                    }
                    var dw = d[i] * scale + L2RegFactor * w[i];
                    if (dw > ClipRange)
                    {
                        dw = ClipRange;
                    }
                    else if (dw < -ClipRange)
                    {
                        dw = -ClipRange;
                    }
                    var hw = h[i] * _decayRate + dw * dw * _keepRate;
                    h[i] = hw;
                    var hw2 = h2[i] * _decayRate2 + dw * _keepRate2;
                    h2[i] = hw2;
                    w[i] -= LearningRate * (hw2 / bc2) / ((float)Math.Sqrt(hw / bc1) + _eps);
                    d[i] = 0;
                }


            }
        }
    }
}