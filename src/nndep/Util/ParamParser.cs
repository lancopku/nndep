using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace nndep.Common
{
    //static class ParamParser
    //{
    //    public static int ParseInt(string value, int def)
    //    {
    //        int res;
    //        return int.TryParse(value, out res) ? res : def;
    //    }

    //    public static bool ParseBool(string value, bool def)
    //    {
    //        value = value.Trim();
    //        if (value.Equals("true", StringComparison.OrdinalIgnoreCase) || value == "1")
    //        {
    //            return true;
    //        }
    //        if (value.Equals("false", StringComparison.OrdinalIgnoreCase) || value == "0")
    //        {
    //            return false;
    //        }
    //        return def;
    //    }

    //    public static double ParseDouble(string value, double def)
    //    {
    //        double res;
    //        return double.TryParse(value, out res) ? res : def;
    //    }

    //    public static float ParseFloat(string value, float def)
    //    {
    //        float res;
    //        return float.TryParse(value, out res) ? res : def;
    //    }

    //    public static string ParsePath(string value, string def)
    //    {
    //        if (value == "null")
    //        {
    //            return def;
    //        }
    //        return File.Exists(value) ? value : def;
    //    }

    //    public static T ParseEnum<T>(string value, T def)
    //        where T : struct
    //    {
    //        T res;
    //        return TryParse(value, out res) ? res : def;
    //    }


    //    private static bool TryParse<T>(string value, out T result)
    //        where T : struct
    //    {
    //        return CreateEnumDictionary<T>().TryGetValue(value.Trim(), out result);
    //    }

    //    private static Dictionary<string, T> CreateEnumDictionary<T>()
    //    {
    //        return Enum.GetValues(typeof(T))
    //            .Cast<T>()
    //            .ToDictionary(value => value.ToString(), value => value, StringComparer.OrdinalIgnoreCase);
    //    }
    //}
}
