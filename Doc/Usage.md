This document shows how to use the causal language model API for text generation.

### Use CausalLMPipeline to generate text

`CausalLMPipeline` provides the most vanilla way to generate text from a language model, which means the prompt will be fed into the model as is, without applying any chat template.

```C#
var pathToPhi3 = "path/to/phi3";
var tokenizer = LLama2Tokenizer.FromPretrained(pathToPhi3);
var phi3CausalModel = Phi3ForCasualLM.FromPretrained(pathToPhi3);

CausalLMPipeline<LLama2Tokenizer, Phi3ForCasualLM> pipeline = new CausalLMPipeline(tokenizer, phi3CausalModel);

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
CausalLMPipeline<LLama2Tokenizer, Phi3ForCasualLM> pipeline = new CausalLMPipeline(tokenizer, phi3CausalModel);
var kernel = Kernel.CreateBuilder()
    // the type of the tokenizer and the model are explicitly specified
    // here for clarity, but the compiler can infer them
    // The typed pipeline prevent developers from passing an arbitrary CausalLMPipeline
    .AddPhi3AsChatCompletionService<LLama2Tokenizer, Phi3ForCasualLM>(pipeline)
    .Build();
```

### Consume model from AutoGen
Similarly, developers would also like to consume the language model like agent.
```C#
var pathToPhi3 = "path/to/phi3";
var tokenizer = LLama2Tokenizer.FromPretrained(pathToPhi3);
var phi3CausalModel = Phi3ForCasualLM.FromPretrained(pathToPhi3);
var pipeline = new CausalLMPipeline(tokenizer, phi3CausalModel);
var agent = new Phi3MiniAgent(pipeline, name: "assistant");

var reply = await agent.SendAsync("Tell me a joke");
```

### Consume model like an OpenAI chat completion service

> [!NOTE]
> This feature is very useful for evaluation and benchmarking. Because most of the benchmarking frameworks are implemented in python, but support consuming openai-like api.

If the model is deployed as a service, developers can consume the model similar to OpenAI chat completion service.
```C#
// server.cs
var pathToPhi3 = "path/to/phi3";
var tokenizer = LLama2Tokenizer.FromPretrained(pathToPhi3);
var phi3CausalModel = Phi3ForCasualLM.FromPretrained(pathToPhi3);
var pipeline = new CausalLMPipeline(tokenizer, phi3CausalModel);
var agent = new Phi3MiniAgent(pipeline, name: "assistant");

// AutoGen.Net allows you to run the agent as an OpenAI chat completion endpoint
var host = Host.CreateDefaultBuilder()
    .ConfigureWebHostDefaults(app =>
    {
        app.UseAgentAsOpenAIChatCompletionEndpoint(agent);
    })
    .Build();

await host.RunAsync();
```

On the client side, the consumption code will be no dfferent from consuming an openai chat completion service.