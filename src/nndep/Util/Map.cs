using System;
using System.Collections.Generic;
using System.Linq;

namespace nndep.Util
{
	public class Map<T>
	{
		private Dictionary<T, int> _id;
		private List<T> _known;

		public Map(IEnumerable<T> set, IEnumerable<T> extra, int cutoff)
		{
			if (cutoff <= 0)
			{
				throw new ArgumentException("cutoff less than 1", nameof(cutoff));
			}
			var freq = new Dictionary<T, int>();
			foreach (var item in set)
			{
				if (freq.ContainsKey(item))
				{
					freq[item] += 1;
				}
				else
				{
					freq[item] = 1;
				}
			}
			_known = extra.Concat(freq.Where(pair => pair.Value >= cutoff).Select(pair => pair.Key)).Distinct().ToList();
			_id = new Dictionary<T, int>(_known.Count);
			for (var i = 0; i < _known.Count; i++)
			{
				_id[_known[i]] = i + 1;
			}
		}

		public Map(IEnumerable<T> set, IEnumerable<T> extra)
		{
			_known = extra.Concat(set.Distinct()).Distinct().ToList();
			_id = new Dictionary<T, int>(_known.Count);
			for (var i = 0; i < _known.Count; i++)
			{
				_id[_known[i]] = i + 1;
			}
		}

		public int Count => _known.Count;
		public int MaxIdx => _known.Count;

		public int this[T key]
		{
			get { return _id.ContainsKey(key) ? _id[key] : 0; }
		}

		public T this[int id]
		{
			get
			{
				if (id == 0)
				{
					return default(T);
				}
				return _known[id - 1];
			}
		}
	}
}