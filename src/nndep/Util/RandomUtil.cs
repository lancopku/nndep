using System;

namespace nndep.Util
{
	public  class RandomNumberGenerator
	{
		private  readonly Random _random;

	    public RandomNumberGenerator(int seed)
	    {
	        if (seed < 0)
	        {
	            _random = new Random();
	        }
	        else
	        {
	            _random = new Random(seed);
	        }
	    }

		public  double GetDouble(float lowerBound, float upperBound)
		{
			return _random.NextDouble() * (upperBound - lowerBound) + lowerBound;
		}

		public  double GetDouble()
		{
			return _random.NextDouble();
		}

		public  float GetFloat(float lowerBound, float upperBound)
		{
			return (float) _random.NextDouble() * (upperBound - lowerBound) + lowerBound;
		}

	    public float GetFloat()
	    {
	        return (float)_random.NextDouble();
	    }

        public  int GetInt(float lowerBound, float upperBound)
		{
			return (int) (_random.NextDouble() * (upperBound - lowerBound) + lowerBound);
		}

		public  int GetIntExclusive(int upperBound)
		{
			return _random.Next(upperBound);
		}

		public  int GetIntExclusive(int lowerBound, int upperBound)
		{
			return _random.Next(lowerBound, upperBound);
		}

		public  float GetNormal(float mean, float stddev)
		{
			var u1 = (float)_random.NextDouble();
			var u2 = (float)_random.NextDouble();
			var randomStdNormal = (float)Math.Sqrt(-2.0 * Math.Log(u1)) * (float)Math.Sin(2.0 * Math.PI * u2);
			return mean + stddev * randomStdNormal;
		}
	}
}