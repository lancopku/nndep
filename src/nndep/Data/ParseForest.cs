using System.Collections.Generic;

namespace nndep.Data
{
	class ParseForest
	{
		private List<Entry> _roots;

		public ParseForest()
		{
			_roots = new List<Entry>();
		}

		public ParseForest(List<Entry> sent, bool reserve)
		{
			_roots = new List<Entry>();
			foreach (var root in sent)
			{
				root.PredParId = -1;
				root.PredRelation = null;
                root.Repr = reserve ? root.Repr : null;
				_roots.Add(root);
			}
		}

		public int Count => _roots.Count;

		public Entry Back()
		{
			if (_roots.Count == 0)
			{
				return null;
			}
			return _roots[_roots.Count - 1];
		}

		public Entry Back(int k)
		{
			if (k >= _roots.Count)
			{
				return null;
			}
			return _roots[_roots.Count - 1 - k];
		}

		public Entry Front()
		{
			if (_roots.Count == 0)
			{
				return null;
			}
			return _roots[0];
		}

		public Entry Front(int k)
		{
			if (k >= _roots.Count)
			{
				return null;
			}
			return _roots[k];
		}

		public Entry PopBack()
		{
			if (_roots.Count == 0)
			{
				return null;
			}
			var end = _roots.Count - 1;
			var ent = _roots[end];
			_roots.RemoveAt(end);
			return ent;
		}

		public Entry PopFront()
		{
			if (_roots.Count == 0)
			{
				return null;
			}
			var ent = _roots[0];
			_roots.RemoveAt(0);
			return ent;
		}

		public void PushBack(Entry ent)
		{
			_roots.Add(ent);
		}

		public void PushFront(Entry ent)
		{
			_roots.Insert(0, ent);
		}
	}
}