using Phi.Module;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;

namespace Phi;

//input_ids: torch.LongTensor = None,
//        attention_mask: Optional[torch.Tensor] = None,
//        position_ids: Optional[torch.LongTensor] = None,
//        past_key_values: Optional[List[torch.FloatTensor]] = None,
//        inputs_embeds: Optional[torch.FloatTensor] = None,
//        use_cache: Optional[bool] = None,
//        output_attentions: Optional[bool] = None,
//        output_hidden_states: Optional[bool] = None,
//        return_dict: Optional[bool] = None,
public class Phi3ModelInput
{
    public Phi3ModelInput(
        Tensor input_ids,
        Tensor? attention_mask = null,
        Tensor? position_ids = null,
        IKVCache? past_key_values = null,
        Tensor? inputs_embeds = null,
        bool use_cache = false,
        bool output_attentions = false,
        bool output_hidden_states = false)
    {
        this.input_ids = input_ids;
        this.attention_mask = attention_mask;
        this.position_ids = position_ids;
        this.past_key_values = past_key_values;
        this.inputs_embeds = inputs_embeds;
        this.use_cache = use_cache;
        this.output_attentions = output_attentions;
        this.output_hidden_states = output_hidden_states;
    }

    public Tensor input_ids { get; set; }

    public Tensor? attention_mask { get; set; }

    public Tensor? position_ids { get; set; }

    public IKVCache? past_key_values { get; set; }

    public Tensor? inputs_embeds { get; set; }

    public bool use_cache { get; set; }

    public bool output_attentions { get; set; }

    public bool output_hidden_states { get; set; }
}

public class Phi3ModelOutput
{
    public Phi3ModelOutput(
        Tensor last_hidden_state,
        Tensor[]? hidden_states = null,
        Tensor[]? attentions = null,
        IKVCache? past_key_values = null)
    {
        this.last_hidden_state = last_hidden_state;
        this.hidden_states = hidden_states;
        this.attentions = attentions;
        this.past_key_values = past_key_values;
    }

    public Tensor last_hidden_state { get; set; }

    public Tensor[]? hidden_states { get; set; }

    public Tensor[]? attentions { get; set; }

    public IKVCache? past_key_values { get; set; }
}
public class Phi3Model : nn.Module<Phi3ModelInput, Phi3ModelOutput>
{
    private readonly Phi3Config config;
    private readonly int padding_idx;
    private readonly int vocab_size;
    private readonly Embedding embed_tokens;
    private readonly Dropout embed_dropout;
    private readonly ModuleList<Phi3DecoderLayer> layers;
    private readonly Phi3RMSNorm norm;

    public Phi3Model(Phi3Config config)
        : base(nameof(Phi3Model))
    {
        this.config = config;
        this.padding_idx = config.PadTokenId;
        this.vocab_size = config.VocabSize;

        this.embed_tokens = nn.Embedding(config.VocabSize, config.HiddenSize, padding_idx: this.padding_idx, dtype: config.DType);
        this.embed_dropout = nn.Dropout(config.EmbdPdrop);
        this.layers = new ModuleList<Phi3DecoderLayer>();

        for (int i = 0; i < config.NumHiddenLayers; i++)
        {
            this.layers.Add(new Phi3DecoderLayer(config, i));
        }
        this.norm = new Phi3RMSNorm(config.HiddenSize, config.RmsNormEps, config.DType);

        this.RegisterComponents();
    }

    public override Phi3ModelOutput forward(Phi3ModelInput input)
    {
        var output_attentions = input.output_attentions;
        var output_hidden_states = input.output_hidden_states;
        var attention_mask = input.attention_mask;
        Device device;
        var input_ids = input.input_ids;
        var position_ids = input.position_ids;
        var inputs_embeds = input.inputs_embeds;
        int batch_size, seq_length;
        if (input_ids is not null && inputs_embeds is not null)
        {
            throw new ArgumentException("Only one of input_ids or inputs_embeds may be set");
        }
        else if (input_ids is not null)
        {
            batch_size = input_ids.IntShape()[0];
            seq_length = input_ids.IntShape()[1];
            inputs_embeds = this.embed_tokens.forward(input_ids);
            device = input_ids.device;
        }
        else if (inputs_embeds is not null)
        {
            batch_size = inputs_embeds.IntShape()[0];
            seq_length = inputs_embeds.IntShape()[1];
            device = inputs_embeds.device;
        }
        else
        {
            throw new ArgumentException("Either input_ids or inputs_embeds must be set");
        }

        var past_key_values_length = 0;

        if (input.past_key_values is not null)
        {
            past_key_values_length = input.past_key_values.GetUsableLength(seq_length);
        }

        if (position_ids is null)
        {
            position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, device: device);
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length);
        }
        else
        {
            position_ids = ((long)position_ids.view(-1, seq_length));
        }

        if (this.config.AttnImplementation == "flash_attention_2")
        {
            throw new NotImplementedException();
        }
        else
        {
            attention_mask = AttentionMaskConverter.Create4DCasualAttentionMask(attention_mask, [batch_size, seq_length], inputs_embeds.dtype, device, past_key_values_length, this.config.SlidingWindow);
        }

        var hidden_states = inputs_embeds;

        var all_hidden_states = new List<Tensor>();
        var all_attentions = new List<Tensor>();

        foreach(var layer in this.layers)
        {
            if (output_hidden_states)
            {
                all_hidden_states.Add(hidden_states);
            }

            var decoderInput = new Phi3DecoderLayerInput(hidden_states, attention_mask!, position_ids, input.past_key_values, output_attentions);
            var layerOutput = layer.forward(decoderInput);
            hidden_states = layerOutput.hidden_states;
            if (output_attentions && layerOutput.attentions is not null)
            {
                all_attentions.Add(layerOutput.attentions);
            }
        }

        hidden_states = this.norm.forward(hidden_states);
        if (output_hidden_states)
        {
            all_hidden_states.Add(hidden_states);
        }

        return new Phi3ModelOutput(hidden_states, all_hidden_states.ToArray(), all_attentions.ToArray(), input.past_key_values);
    }

}
