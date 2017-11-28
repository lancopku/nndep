using System.Collections.Generic;
using System.Text;
using nndep.Networks.Graph;

namespace nndep.Networks.Opts
{
	abstract class OptBase
	{
		private readonly OptType _type;
		protected readonly float ClipRange;
		protected readonly float L2RegFactor;
		protected readonly float LearningRate;


		protected OptBase(OptType type, float lr, float clipRange, float l2RegFactor)
		{
			ClipRange = clipRange;
			L2RegFactor = l2RegFactor;
			LearningRate = lr;
			_type = type;
		}


		public abstract void Prepare(IEnumerable<Tensor> p);


		public override string ToString()
		{
			var sb = new StringBuilder();
			sb.AppendLine($"Type = {_type}");
			sb.AppendLine($"LearningRate = {LearningRate:g2}");
			sb.AppendLine($"ClipRange = {ClipRange:g2}");
			sb.AppendLine($"L2RegFactor = {L2RegFactor:g2}");
			return sb.ToString();
		}


		public abstract void Update(IEnumerable<Tensor> parameters, float scale);

        public abstract void Update(IEnumerable<Tensor> parameters);
    }
}