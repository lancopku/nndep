using System;
using nndep.Util;

namespace nndep.Networks.Inits
{
	class GlorotNormalInit : IInit
	{
		private readonly float _factor;
		private readonly int _fanIn;
		private readonly int _fanOut;
	    private readonly RandomNumberGenerator _rng;

		public GlorotNormalInit(int fanIn, int fanOut, RandomNumberGenerator rng)
		{
			_fanIn = fanIn;
			_fanOut = fanOut;
			_factor = (float)(Math.Sqrt(6.0f / (fanIn + fanOut)));
		    _rng = rng;
		}

		public float Next()
		{
			return _rng.GetNormal(0, _factor);
		}

		public override string ToString()
		{
			return $"GlorotNormalInit[{_fanIn}->{_fanOut}]";
		}
	}
}