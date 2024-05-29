# What is a causal language model pipeline?

The causal language model pipeline is a utility class which wraps a tokenizer and a causal language model and provides a uniformed interface for various decoding method to generate text. The pipeline is designed to be easy to use and requires only a few lines of code to generate text.

# Contract
```C#
public abstract class CasualLMPipeline
{
    public virtual (
        Tensor, // output token ids [batch_size, sequence_length]
        Tensor // output logits [batch_size, sequence_length, vocab_size]
    ) Generate(
        Tensor inputIds, // input token ids [batch_size, sequence_length]
        Tensor attentionMask, // attention mask [batch_size, sequence_length]
        float temperature = 0.7f,
        float topP = 0.9f,
        int maxLen = 128,
        int[][]? stopTokenSequence = null,
        bool echo = false); // echo the input token ids in the output token ids
}

public CasualLMPipeline<TTokenizer, TCasualLM> : CasualLMPipeline
    where TTokenizer : ITokenizer
    where TCasualLM : nn.Module<CausalLanguageModelInput, CausalLanguageModelOutput>
{
    public CasualLMPipeline<LLama2Tokenizer, Phi3ForCasualLM> Create(LLama2Tokenizer tokenizer, Phi3ForCasualLM model);

}
```

# Usage
```C#
LLama2Tokenizer tokenizer;
Phi3ForCasualLM model;

var pipeline = CausalLMPipeline.Create(tokenizer, model);
var prompt = "Once upon a time";
// top-k sampling
var output = pipeline.Generate(
    prompt: prompt,
    maxLen: 100,
    temperature: 0.7f,
    topP: 0.9f,
    stopSequences: null,
    device: "cuda",
    bos: true, // add bos token to the prompt
    eos: false, // do not add eos token to the prompt
    echo: true // echo the prompt in the generated text
);
```

# Sampling methods
The `CaualLMPipeline` provides a uniformed interface for various decoding methods to generate text. This saves our effort to implement different decoding methods for each model.

## Sampling
```C#
public virtual (
        Tensor, // output token ids [batch_size, sequence_length]
        Tensor // output logits [batch_size, sequence_length, vocab_size]
    ) Generate(
        Tensor inputIds, // input token ids [batch_size, sequence_length]
        Tensor attentionMask, // attention mask [batch_size, sequence_length]
        float temperature = 0.7f,
        float topP = 0.9f,
        int maxLen = 128,
        int[][]? stopTokenSequence = null,
        bool echo = false); // echo the input token ids in the output token ids
```

>[!NOTE]
> The Greedy search and beam search are not implemented in the pipeline yet. They will be added in the future.

## Greedy Search
```C#
public (
    Tensor, // output token ids [batch_size, sequence_length]
    Tensor // output logits [batch_size, sequence_length, vocab_size]
) GreedySearch(
    Tensor inputIds, // input token ids [batch_size, sequence_length]
    Tensor attentionMask, // attention mask [batch_size, sequence_length]
    int maxLen = 128,
    int[][]? stopTokenSequence = null,
    bool echo = false); // echo the input token ids in the output token ids
```

## Beam Search
```C#
public (
    Tensor, // output token ids [batch_size, sequence_length]
    Tensor // output logits [batch_size, sequence_length, vocab_size]
) BeamSearch(
    Tensor inputIds, // input token ids [batch_size, sequence_length]
    Tensor attentionMask, // attention mask [batch_size, sequence_length]
    int maxLen = 128,
    int[][]? stopTokenSequence = null,
    int beamSize = 5,
    float lengthPenalty = 1.0f,
    bool echo = false); // echo the input token ids in the output token ids
```

## The extension method for `CausalLMPipeline`

The extension `Generate` method provides a even-easier way to generate text without the necessary to generate the input tensor. The method takes a prompt string and other optional parameters to generate text.

```C#
public static string Generate(
    this CasualLMPipeline pipeline,
    string prompt,
    int maxLen = 128,
    float temperature = 0.7f,
    float topP = 0.9f,
    string[]? stopSequences = null,
    string device = "cpu",
    bool bos = true,
    bool eos = false,
    bool echo = false)
```