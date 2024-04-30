using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Phi;

public interface IKVCache : IDictionary<int, (Tensor, Tensor)>
{
    public (Tensor, Tensor) UpdateKVCache(Tensor key, Tensor value, int layer_idx);

    public int GetSeqLen(int layerIndex = 0);

    public int? GetMaxLength();

    public int GetUsableLength(int newSeqLen, int layerIndex = 0);
}

public class DynamicKVCache : Dictionary<int, (Tensor, Tensor)>, IKVCache
{
    public DynamicKVCache()
    {
    }

    public (Tensor, Tensor) UpdateKVCache(Tensor key, Tensor value, int layer_idx)
    {
        if (this.ContainsKey(layer_idx))
        {
            var (oldKey, oldValue) = this[layer_idx];
            oldKey = torch.cat([oldKey, key], -2);
            oldValue = torch.cat([oldValue, value], -2);
            this[layer_idx] = (oldKey, oldValue);
        }
        else
        {
            this.Add(layer_idx, (key, value));
        }

        return this[layer_idx];
    }

    public int GetSeqLen(int layerIndex = 0)
    {
        if (this.TryGetValue(layerIndex, out var kv))
        {
            return kv.Item1.IntShape()[^2];
        }

        return 0;
    }

    public int? GetMaxLength()
    {
        return null;
    }

    public int GetUsableLength(int newSeqLen, int layerIndex = 0)
    {
        var max_length = this.GetMaxLength();
        var previousSeqLen = this.GetSeqLen(layerIndex);

        if (max_length.HasValue && previousSeqLen + newSeqLen > max_length.Value)
        {
            return max_length.Value - previousSeqLen;
        }

        return previousSeqLen;
    }
}
