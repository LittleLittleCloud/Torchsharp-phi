using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Phi;

public interface IKVCache : IDictionary<int, (Tensor, Tensor)>, IDisposable
{
    public (Tensor, Tensor) UpdateKVCache(Tensor key, Tensor value, int layer_idx);

    public int GetSeqLen(int layerIndex = 0);

    public int? GetMaxLength();

    public int GetUsableLength(int newSeqLen, int layerIndex = 0);
}

public class DynamicKVCache : Dictionary<int, (Tensor, Tensor)>, IKVCache
{
    private readonly DisposeScope disposeScope = new DisposeScope(new DisposeScopeManager());
    public DynamicKVCache()
    {
    }

    public (Tensor, Tensor) UpdateKVCache(Tensor key, Tensor value, int layer_idx)
    {
        if (this.ContainsKey(layer_idx))
        {
            var (oldKey, oldValue) = this[layer_idx];
            oldKey.DetachFromDisposeScope();
            oldValue.DetachFromDisposeScope();

            var newKey = torch.cat([oldKey, key], -2).MoveToOtherDisposeScope(this.disposeScope);
            var newValue = torch.cat([oldValue, value], -2).MoveToOtherDisposeScope(this.disposeScope);

            oldKey.Dispose();
            oldValue.Dispose();

            this[layer_idx] = (newKey, newValue);
        }
        else
        {
            this.Add(layer_idx, (key.MoveToOtherDisposeScope(this.disposeScope), value.MoveToOtherDisposeScope(this.disposeScope)));
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

    public void Dispose()
    {
        this.disposeScope.Dispose();
    }
}
