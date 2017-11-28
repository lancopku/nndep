using nndep.Data;
using nndep.Util;
using System;

namespace nndep.ArcStd
{
	class ArcStandardConfig
	{
		public enum Op
		{
			shift,
			larc,
			rarc
		}

		private readonly Sent _sent;
		public readonly ParseForest Buffer;
		public readonly ParseForest Stack;
	    private readonly RandomNumberGenerator _rng;

		public ArcStandardConfig(Sent sent, bool reserve, RandomNumberGenerator rng)
		{
			_sent = sent;
			Stack = new ParseForest();
			Stack.PushBack(Entry.Root());
			Buffer = new ParseForest(sent, reserve);
            Apply(Op.shift, null);
		    _rng = rng;
		}


		public void Apply(Op op, string relation)
		{
			switch (op)
			{
				case Op.shift:
                    var next = Buffer.PopFront();
                    Checker.Asserts<InvalidOperationException>(next != null, "cannot shift null");
					Stack.PushBack(next);
					break;
				case Op.larc:
				{
					var right = Stack.PopBack();
					var left = Stack.PopBack();
					left.SetHead(right, relation);
					Stack.PushBack(right);
				}
					break;
				case Op.rarc:
				{
					var right = Stack.PopBack();
					var left = Stack.Back();
					right.SetHead(left, relation);
				}
					break;
			}
		}


        public bool CanApply(Op op, string label)
		{
			var nBuffer = Buffer.Count;
			var nStack = Stack.Count;
			switch (op)
			{
				case Op.shift:
					return nBuffer > 0;
				case Op.larc:
                    {
                        return nStack > 2;
                    }
				case Op.rarc:
                    {
                        return nStack > 2 || (nStack == 2 && nBuffer == 0);
                    }
				default:
					return false;
			}
		}

        private bool HasUnattachedChild(int k)
        {
            foreach (var ent in _sent)
            {
                if (ent.Pid == k && ent.PredParId != k)
                {
                    return true;
                }
            }
            return false;
        }

        private bool HasRightUnattachedChild(int k)
        {
            foreach(var ent in _sent)
            {
                if(ent.Id>k && ent.Pid ==k && ent.PredParId != k)
                {
                    return true;
                }
            }
            return false;
        }

        //private bool HasLeftUnattachedChild(int k)
        //{
        //    foreach (var ent in _sent)
        //    {
        //        if (ent.Id < k && ent.Pid == k && ent.PredParId != k)
        //        {
        //            return true;
        //        }
        //    }
        //    return false;
        //}

        public enum OracleType {standard, hybrid}


        public Op GetOracle(OracleType type, out float[] oracles, out string label)
        {
            oracles = new float[3];
            label = null;
            var nBuffer = Buffer.Count;
            var nStack = Stack.Count;


            if (nStack > 2) // can arc?
            {
                oracles[(int)Op.larc] = 0;
                oracles[(int)Op.rarc] = 0;
            }
            else{
                // larc is not okay
                oracles[(int)Op.larc] = -1;
                if (nStack == 2 && nBuffer == 0) // rarc of root?
                {
                    //rarc of root
                    oracles[(int)Op.rarc] = 0;
                }
                else
                {
                    // rarc is not okay
                    oracles[(int)Op.rarc] = -1;
                }
            }

            // can shift?
            if (nBuffer > 0)
            {
                oracles[(int)Op.shift] = 0;
            }
            else
            {
                oracles[(int)Op.shift] = -1;
            }


            var right = Stack.Back();
            var left = Stack.Back(1);
            
            // left?
            if (left != null && left.Pid == right.Id && !HasUnattachedChild(left.Id))
            {
                // left is okay
                if (type == OracleType.standard)
                {
                    // only larc
                    oracles[(int)Op.larc] = 1;
                    label = left.Relation;
                    return Op.larc;
                }
                else
                {
                    // shift?
                    if (HasRightUnattachedChild(right.Id))
                    {
                        // shift is also okay
                        if (type == OracleType.hybrid)
                        {
                            // larc and shift
                            oracles[(int)Op.larc] = 0.5f;
                            oracles[(int)Op.shift] = 0.5f;
                            label = left.Relation;
                            var n = _rng.GetDouble();
                            if (n < 0.5)
                            {
                                return Op.larc;
                            }
                            else
                            {
                                return Op.shift;
                            }
                        }
                        else
                        {
                            // shift
                            oracles[(int)Op.shift] = 1;
                            label = null;
                            return Op.shift;
                        }
                    }
                    else
                    {
                        // only larc
                        oracles[(int)Op.larc] = 1;
                        label = left.Relation;
                        return Op.larc;
                    }
                }
            }
            
            // rarc?
            if (left != null && (left.Id > 0 || (left.Id == 0 && Stack.Count == 2)) && right.Pid == left.Id &&
                !HasUnattachedChild(right.Id))
            {
                oracles[(int)Op.rarc] = 1.0f;
                label = right.Relation;
				return Op.rarc;
			}

            // only shift
            label = null;
            oracles[(int)Op.shift] = 1.0f;
			return Op.shift;
		}

        public bool IsTerminal()
		{
			return Stack.Count == 1 && Buffer.Count == 0;
		}
	}
}