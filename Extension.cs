using TorchSharp;
using static TorchSharp.torch;

public static class Extension
{
    public static void Peek(this Tensor tensor, string id, int n = 10)
    {
        var shapeString = string.Join(',', tensor.shape);
        var dataString = string.Join(',', tensor.reshape(-1)[..n].to_type(ScalarType.Float32).data<float>().ToArray());
        var tensor_1d = tensor.reshape(-1);
        var tensor_index = torch.arange(tensor_1d.shape[0], dtype: ScalarType.Float32).to(tensor_1d.device).sqrt();
        var avg = (tensor_1d * tensor_index).sum() / tensor_1d.sum();
        Console.WriteLine($"{id}: sum: {avg.ToSingle()}  dtype: {tensor.dtype} shape: [{shapeString}] device: {tensor.device}");
    }

    public static void Peek(this nn.Module model)
    {
        var state_dict = model.state_dict();
        // preview state_dict
        foreach (var (key, value) in state_dict)
        {
            Console.WriteLine($"key: {key} value: {value}");
        }
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

            TensorExtensionMethods.Load(ref tensor, reader, skip: false);
            
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