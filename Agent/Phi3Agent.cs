using AutoGen.Core;
using Phi.Pipeline;
using System.Text;

namespace Phi.Agent;

public class Phi3Agent : IAgent
{
    private const char newline = '\n';
    private readonly CasualLMPipeline pipeline;
    public Phi3Agent(CasualLMPipeline pipeline, string name)
    {
        this.Name = name;
        this.pipeline = pipeline;
    }

    public string Name { get; }

    public async Task<IMessage> GenerateReplyAsync(IEnumerable<IMessage> messages, GenerateReplyOptions? options = null, CancellationToken cancellationToken = default)
    {
        var availableRoles = new[] { Role.System, Role.User, Role.Assistant };
        if (messages.Any(m => m.GetContent() is null))
        {
            return new TextMessage(Role.Assistant, "Please provide a message with content.", from: this.Name);
        }

        if (messages.Any(m => m.GetRole() is null || availableRoles.Contains(m.GetRole()!.Value) == false))
        {
            return new TextMessage(Role.Assistant, "Please provide a message with a valid role. The valid roles are System, User, and Assistant.", from: this.Name);
        }

        // construct template based on instruction from
        // https://huggingface.co/microsoft/Phi-3-mini-128k-instruct#chat-format

        var sb = new StringBuilder();
        foreach (var message in messages)
        {
            var role = message.GetRole()!.Value;
            var content = message.GetContent()!;
            sb.Append(message switch
            {
                _ when message.GetRole() == Role.System => $"<|system|>{newline}{content}<|end|>{newline}",
                _ when message.GetRole() == Role.User => $"<|user|>{newline}{content}<|end|>{newline}",
                _ when message.GetRole() == Role.Assistant => $"<|assistant|>{newline}{content}<|end|>{newline}",
                _ => throw new InvalidOperationException("Invalid role.")
            });
        }

        sb.Append("<|assistant|>");
        var input = sb.ToString();

        var maxLen = options?.MaxToken ?? 1024;
        var temperature = options?.Temperature ?? 0.7f;
        var stopTokenSequence = options?.StopSequence ?? [];
        stopTokenSequence = stopTokenSequence.Append("<|end|>").ToArray();

        Console.WriteLine("prompt: " + input);
        var output = pipeline.Generate(
            input,
            maxLen: maxLen,
            temperature: temperature,
            stopSequences: stopTokenSequence,
            device: "cuda");

        return new TextMessage(Role.Assistant, output, from: this.Name);
    }
}
