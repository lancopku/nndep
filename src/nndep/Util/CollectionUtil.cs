using System;
using System.Collections.Generic;
using System.Linq;

namespace nndep.Util
{
	public static class CollectionUtil
	{
		public static List<List<T>> PartitionIntoFolds<T>(List<T> values, int foldCount)
		{
			if (foldCount == 0)
			{
				return new List<List<T>> {values};
			}
			var folds = new List<List<T>>();
			var count = values.Count;
			var foldSize = count / foldCount;
			var remainder = count % foldCount;
			var start = 0;
			for (var foldNum = 0; foldNum < foldCount; foldNum++)
			{
				var size = foldSize;
				if (foldNum < remainder)
				{
					size++;
				}
				folds.Add(values.GetRange(start, size));
				start += size;
			}
			return folds;
		}

		public static void Shuffle<T>(this IList<T> list, RandomNumberGenerator rng)
		{
			for (var i = 0; i < list.Count; i++)
			{
				list.Swap(i, rng.GetIntExclusive(i, list.Count));
			}
		}

		public static void Swap<T>(this IList<T> list, int i, int j)
		{
			var temp = list[i];
			list[i] = list[j];
			list[j] = temp;
		}

        public static List<float> ToStdNorm(this IList<float> values)
        {
            var avg = values.Average();
            var cnt = values.Count;
            var std = (float)Math.Sqrt(values.Sum(d => (d - avg) * (d - avg)));
            return values.Select(x => (x-avg) / std).ToList();
        }
    }
}