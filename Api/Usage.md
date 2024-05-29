This document shows how to use the causal language model API for text generation.

### Use CausalLMPipeline to generate text

`CausalLMPipeline` provides the most vanilla way to generate text from a language model, which means the prompt will be fed into the model as is, without applying any chat template.

```C#
var pathToPhi3 = "path/to/phi3";
var tokenizer = LLama2Tokenizer.FromPretrained(pathToPhi3);
var phi3CausalModel = Phi3ForCasualLM.FromPretrained(pathToPhi3);

var pipeline = new CausalLMPipeline(tokenizer, phi3CausalModel);

var prompt = "<|user|>Once upon a time<|end|><assistant>";
var output = pipeline.Generate(
    prompt: prompt,

    maxLen: 100);
```

### Consume model from semantic kernel
In most cases, developers would like to consume the model in a uniformed way. In this case, we can provide an extension method to semantic kernel which adds CausalLMPipeline as `ChatCompletionService`

```C#
var pathToPhi3 = "path/to/phi3";
var tokenizer = LLama2Tokenizer.FromPretrained(pathToPhi3);
var phi3CausalModel = Phi3ForCasualLM.FromPretrained(pathToPhi3);
var pipeline = new CausalLMPipeline(tokenizer, phi3CausalModel);
var kernel = Kernel.CreateBuilder()
    .AddCausalLMPipelineAsChatCompletionService(pipeline)
    .Build();
```