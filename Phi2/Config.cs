using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using static TorchSharp.torch;

public class Phi2Config
{
    [JsonPropertyName("vocab_size")]
    public int VocabSize { get; set; } = 51200;

    [JsonPropertyName("hidden_size")]
    public int HiddenSize { get; set; } = 2048;

    [JsonPropertyName("intermediate_size")]
    public int IntermediateSize { get; set; } = 8192;

    [JsonPropertyName("num_hidden_layers")]
    public int NumHiddenLayers { get; set; } = 24;

    [JsonPropertyName("num_attention_heads")]
    public int NumAttentionHeads { get; set; } = 32;

    [JsonPropertyName("num_key_value_heads")]
    public int? NumKeyValueHeads { get; set; } = null;

    [JsonPropertyName("resid_pdrop")]
    public double ResidPdrop { get; set; } = 0.0;

    [JsonPropertyName("embd_pdrop")]
    public double EmbdPdrop { get; set; } = 0.0;

    [JsonPropertyName("attention_dropout")]
    public double AttentionDropout { get; set; } = 0.0;

    [JsonPropertyName("hidden_act")]
    public string HiddenAct { get; set; } = "gelu_new";

    [JsonPropertyName("max_position_embeddings")]
    public int MaxPositionEmbeddings { get; set; } = 2048;

    [JsonPropertyName("initializer_range")]
    public double InitializerRange { get; set; } = 0.02;

    [JsonPropertyName("layer_norm_eps")]
    public double LayerNormEps { get; set; } = 1e-5;

    [JsonPropertyName("use_cache")]
    public bool UseCache { get; set; } = true;

    [JsonPropertyName("tie_word_embeddings")]
    public bool TieWordEmbeddings { get; set; } = false;

    [JsonPropertyName("rope_theta")]
    public double RopeTheta { get; set; } = 10000.0;

    // [JsonPropertyName("rope_scaling")]
    // public double? RopeScaling { get; set; } = null;

    [JsonPropertyName("partial_rotary_factor")]
    public double PartialRotaryFactor { get; set; } = 0.5;

    [JsonPropertyName("qk_layernorm")]
    public bool QkLayernorm { get; set; } = false;

    [JsonPropertyName("bos_token_id")]
    public int BosTokenId { get; set; } = 1;

    [JsonPropertyName("eos_token_id")]
    public int EosTokenId { get; set; } = 2;

    public ScalarType Dtype { get; set; } = ScalarType.Float32;
}
