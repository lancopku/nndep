using System;
using System.Collections.Generic;
using System.Text;
using nndep.Networks.Graph;

namespace nndep.Networks.Opts
{
	class AdaGrad : OptBase
	{
		private readonly float _eps;


		public AdaGrad(float lr, float l2RegFactor, float clipRange, float eps)
			: base(OptType.adagrad, lr, clipRange, l2RegFactor)
		{
			_eps = eps;
		}

		public override void Prepare(IEnumerable<Tensor> p)
		{
			foreach (var t in p)
			{
				t.InitParam(false, true);
			}
		}


		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.Append(base.ToString());
			sb.AppendLine($"Eps = {_eps:g2}");
			return sb.ToString();
		}

		public override void Update(IEnumerable<Tensor> parameters, float scale = 1f)
		{
			foreach (var p in parameters)
			{
				var w = p.W.Storage;
				var d = p.Grad.Storage;
				var h = p.VarMom.Storage;
				for (var i = 0; i < w.Length; i++)
				{
					var dw = d[i] * scale + L2RegFactor * w[i];
					if (dw > ClipRange)
					{
						dw = ClipRange;
					}
					else if (dw < -ClipRange)
					{
						dw = -ClipRange;
					}
					var hw = h[i] + dw * dw;
					h[i] = hw;
					w[i] -= dw * (LearningRate / (float)Math.Sqrt(hw + _eps));
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