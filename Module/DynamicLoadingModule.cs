using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp;
using static TorchSharp.torch;

namespace Phi.Module;

public class DynamicLoadingModule<T, T1, TResult> : torch.nn.Module<T1, TResult>, IDynamicLoadModule
    where T : nn.Module<T1, TResult>
    where T1 : Tensor
{
    private readonly T model;

    public DynamicLoadingModule(T model)
        : base(model.GetName())
    {
        this.model = model;
        this.RegisterComponents();
    }

    public static DynamicLoadingModule<T, T1, TResult> CreateFromModel(T model)
    {
        return new DynamicLoadingModule<T, T1, TResult>(model);
    }

    public Action<nn.Module>? LoadToDeviceFunc { get; set; } = null;
    public Action<nn.Module>? UnloadFromDeviceFunc { get; set; } = null;

    public override TResult forward(T1 input)
    {
        if (LoadToDeviceFunc != null)
        {
            LoadToDeviceFunc(this);
        }

        var output = this.model.forward(input);

        if (UnloadFromDeviceFunc != null)
        {
            UnloadFromDeviceFunc(this);
        }

        return output;
    }
}
