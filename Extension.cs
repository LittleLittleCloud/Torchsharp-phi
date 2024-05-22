using Phi.Pipeline;
using System.Text;
using TorchSharp;
using static TorchSharp.torch;

public static class Extension
{
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
    {
        var inputIds = pipeline.Tokenizer.Encode(prompt, bos, eos);
        var inputTensor = torch.tensor(inputIds.ToArray(), dtype: ScalarType.Int64, device: device).unsqueeze(0);
        var attentionMask = torch.ones_like(inputTensor);
        var stopTokenIds = stopSequences == null ? [[ pipeline.Tokenizer.EosId ]] : stopSequences.Select(x => pipeline.Tokenizer.Encode(x, false, false)).ToArray();
        (var token, var _) = pipeline.Generate(inputTensor, attentionMask, temperature: temperature, maxLen: maxLen, topP: topP, stopTokenSequence: stopTokenIds, echo: echo);

        var tokenIds = token[0].to_type(ScalarType.Int32).data<int>().ToArray();
        var output = pipeline.Tokenizer.Decode(tokenIds);
        return output;

    }

    public static string Peek(this Tensor tensor, string id, int n = 10)
    {
        var device = tensor.device;
        var dtype = tensor.dtype;
        // if type is fp16, convert to fp32
        if (tensor.dtype == ScalarType.Float16)
        {
            tensor = tensor.to_type(ScalarType.Float32);
        }
        tensor = tensor.cpu();
        var shapeString = string.Join(',', tensor.shape);
        var tensor_1d = tensor.reshape(-1);
        var tensor_index = torch.arange(tensor_1d.shape[0], dtype: ScalarType.Float32).to(tensor_1d.device).sqrt();
        var avg = (tensor_1d * tensor_index).sum();
        avg = avg / tensor_1d.sum();
        // keep four decimal places
        avg = avg.round(4);
        var str = $"{id}: sum: {avg.ToSingle()}  dtype: {dtype} shape: [{shapeString}]";

        Console.WriteLine(str);

        return str;
    }

    public static string Peek(this nn.Module model)
    {
        var sb = new StringBuilder();
        var state_dict = model.state_dict();
        // preview state_dict
        int i = 0;
        foreach (var (key, value) in state_dict.OrderBy(x => x.Key, StringComparer.OrdinalIgnoreCase))
        {
            var str = value.Peek(key);
            sb.AppendLine($"{i}: {str}");
            i++;
        }

        var res = sb.ToString();

        //Console.WriteLine(res);

        return res;
    }

    public static string Peek_Shape(this nn.Module model)
    {
        var sb = new StringBuilder();
        var state_dict = model.state_dict();
        // preview state_dict
        int i = 0;
        foreach (var (key, value) in state_dict.OrderBy(x => x.Key, StringComparer.OrdinalIgnoreCase))
        {
            // shape str: [x, y, z]
            var shapeStr = string.Join(", ", value.shape);
            sb.AppendLine($"{i}: {key} shape: [{shapeStr}]");
            i++;
        }

        var res = sb.ToString();

        Console.WriteLine(res);

        return res;
    }

    public static void LoadStateDict(this Dictionary<string, Tensor> dict, string location)
    {
        using FileStream stream = File.OpenRead(location);
        using BinaryReader reader = new BinaryReader(stream);
        var num = reader.Decode();
        Console.WriteLine($"num: {num}");
        for (int i = 0; i < num; i++)
        {
            var key = reader.ReadString();
            Tensor tensor = dict[key];
            Console.WriteLine($"load key: {key} tensor: {tensor}");

            var originalDevice = tensor.device;
            var originalType = tensor.dtype;
            if (tensor.dtype == ScalarType.BFloat16)
            {
                tensor = tensor.to_type(ScalarType.Float32);
            }

            TensorExtensionMethods.Load(ref tensor!, reader, skip: false);
            
            // convert type to bf16 if type is float
            tensor = tensor!.to_type(originalType);
            dict[key] = tensor;
        }
    }

    //
    // 摘要:
    //     Decode a long value from a binary reader
    //
    // 参数:
    //   reader:
    //     A BinaryReader instance used for input.
    //
    // 返回结果:
    //     The decoded value
    public static long Decode(this BinaryReader reader)
    {
        long num = 0L;
        int num2 = 0;
        while (true)
        {
            long num3 = reader.ReadByte();
            num += (num3 & 0x7F) << num2 * 7;
            if ((num3 & 0x80) == 0L)
            {
                break;
            }

            num2++;
        }

        return num;
    }
}