This document shows how to use the causal language model API for text generation.

## Usage
```C#
var pathToPhi3 = "path/to/phi3";
var tokenizer = LLama2Tokenizer.FromPretrained(pathToPhi3);
var phi3CausalModel = Phi3ForCasualLM.FromPretrained(pathToPhi3);

var pipeline = new CausalLMPipeline(tokenizer, phi3CausalModel);

var prompt = "Once upon a time";
var output = pipeline.Generate(
    prompt: prompt,
    maxLen: 100);
```