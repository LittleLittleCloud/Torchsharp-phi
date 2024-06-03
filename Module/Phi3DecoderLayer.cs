using Phi.Module;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Phi;

public class Phi3DecoderLayerInput
{
    public Phi3DecoderLayerInput(
        Tensor hidden_states,
        Tensor attention_mask,
        Tensor position_ids,
        IKVCache? past_key_value = null,
        bool output_attentions = false)
    {
        this.hidden_states = hidden_states;
        this.attention_mask = attention_mask;
        this.position_ids = position_ids;
        this.past_key_value = past_key_value;
        this.output_attentions = output_attentions;
    }

    public Tensor hidden_states { get; set; }

    public Tensor attention_mask { get; set; }

    public Tensor position_ids { get; set; }

    public IKVCache? past_key_value { get; set; }

    public bool output_attentions { get; set; }
}

public class Phi3DecoderLayerOutput
{
    public Phi3DecoderLayerOutput(
        Tensor hidden_states,
        Tensor? attentions = null,
        IKVCache? past_key_value = null)
    {
        this.hidden_states = hidden_states;
        this.attentions = attentions;
        this.past_key_value = past_key_value;
    }

    public Tensor hidden_states { get; set; }

    public Tensor? attentions { get; set; }

    public IKVCache? past_key_value { get; set; }
}

public class Phi3DecoderLayer : nn.Module<Phi3DecoderLayerInput, Phi3DecoderLayerOutput>, IDynamicLoadModule
{
    private readonly Phi3Config config;
    private readonly nn.Module<Phi3AttentionInput, Phi3AttentionOutput> self_attn;
    private readonly Phi3MLP mlp;
    private readonly Phi3RMSNorm input_layernorm;
    private readonly Dropout resid_attn_dropout;
    private readonly Dropout resid_mlp_dropout;
    private readonly Phi3RMSNorm post_attention_layernorm;

    public Phi3DecoderLayer(Phi3Config config, int layer_idx)
        : base(nameof(Phi3DecoderLayer))
    {
        this.config = config;
        if (config.AttnImplementation == "eager")
        {
            this.self_attn = new Phi3Attention(config, layer_idx);
        }
        else
        {
            throw new NotImplementedException();
        }

        this.mlp = new Phi3MLP(config);
        this.input_layernorm = new Phi3RMSNorm(config.HiddenSize, config.RmsNormEps, config.DType);

        this.resid_attn_dropout = nn.Dropout(config.ResidPdrop);
        this.resid_mlp_dropout = nn.Dropout(config.ResidPdrop);
        this.post_attention_layernorm = new Phi3RMSNorm(config.HiddenSize, config.RmsNormEps, config.DType);
    }

    public Action<nn.Module>? LoadToDeviceFunc { get; set; }

    public Action<nn.Module>? UnloadFromDeviceFunc { get; set; }

    public override Phi3DecoderLayerOutput forward(Phi3DecoderLayerInput input)
    {
        if (LoadToDeviceFunc != null)
        {
            LoadToDeviceFunc(this);
        }
        using var _ = NewDisposeScope();
        var hidden_states = input.hidden_states;
        var residual = input.hidden_states;
        hidden_states = this.input_layernorm.forward(hidden_states);

        var attentionInput = new Phi3AttentionInput(hidden_states, input.position_ids, input.attention_mask, input.past_key_value, input.output_attentions);
        var output = this.self_attn.forward(attentionInput);
        var attn_outputs = output.hidden_states;
        var self_attn_weights = output.attentions;
        var present_key_value = output.cache;
        
        hidden_states = residual + this.resid_attn_dropout.forward(attn_outputs);
        residual = hidden_states;
        hidden_states = this.post_attention_layernorm.forward(hidden_states);
        hidden_states = this.mlp.forward(hidden_states);
        hidden_states = residual + this.resid_mlp_dropout.forward(hidden_states);

        if (UnloadFromDeviceFunc != null)
        {
            UnloadFromDeviceFunc(this);
        }
        return new Phi3DecoderLayerOutput(hidden_states.MoveToOuterDisposeScope(), self_attn_weights?.MoveToOuterDisposeScope(), present_key_value);
    }
}
