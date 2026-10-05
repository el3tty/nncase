using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Nncase.IR;
using SkiaSharp;

namespace Nncase.Quantization;

/// <summary>
/// <see cref="ICalibrationDatasetProvider"/> for standard image dataset folders (JPG, PNG, BMP).
/// Images already at the model input resolution are used as is; others are letterboxed (gray 114 padding).
/// Output is RGB in NCHW layout, in the element type of the model input:
/// <c>uint8</c> input gives raw 0..255 bytes, <c>float32</c> input gives values normalized to 0..1.
/// Decoding and resizing use SkiaSharp (MIT).
/// </summary>
public sealed class ImageCalibrationDatasetProvider : ICalibrationDatasetProvider
{
    private const byte LetterboxGray = 114;

    private static readonly string[] SupportedExtensions = { ".jpg", ".jpeg", ".png", ".bmp" };

    public ImageCalibrationDatasetProvider(IReadOnlyList<Var> vars, string datasetPath)
    {
        Trace.Assert(Directory.Exists(datasetPath), "The dataset folder path must be valid!");
        Trace.Assert(vars.Count == 1, "Currently ImageProvider supports single-input models (e.g. YOLO).");

        var inputVar = vars[0];
        Trace.Assert(inputVar.CheckedType is TensorType, "Input var must be a TensorType.");

        var tensorType = (TensorType)inputVar.CheckedType;
        Trace.Assert(tensorType.Shape.IsFixed, "Model input shape must be fixed for image dataset provider.");

        // Expected NCHW shape, eg [1, 3, 640, 640] or [1, 3, 320, 320]
        int[] shape = tensorType.Shape.ToValueArray();
        Trace.Assert(shape.Length == 4 && shape[1] == 3, "Expected NCHW tensor layout with 3 channels.");

        int targetWidth = shape[3];
        int targetHeight = shape[2];

        var dataType = tensorType.DType;
        bool isUInt8 = dataType == DataTypes.UInt8;
        Trace.Assert(isUInt8 || dataType == DataTypes.Float32, $"Unsupported model input type '{dataType}'. Only uint8 and float32 are supported.");

        var imageFiles = Directory.EnumerateFiles(datasetPath)
            .Where(f => SupportedExtensions.Contains(Path.GetExtension(f).ToLowerInvariant()))
            .OrderBy(f => f, StringComparer.Ordinal)
            .ToList();

        Trace.Assert(imageFiles.Count > 0, $"No supported images found in directory: {datasetPath}");

        Count = imageFiles.Count;

        Samples = imageFiles.Select(filePath =>
        {
            var values = new Dictionary<Var, IValue>();
            byte[] planar = LoadImageToNchwUInt8(filePath, targetWidth, targetHeight);

            Tensor tensor;
            if (isUInt8)
            {
                tensor = Tensor.From<byte>(planar, shape);
            }
            else
            {
                var floats = new float[planar.Length];
                for (int i = 0; i < planar.Length; i++)
                {
                    floats[i] = planar[i] / 255.0f;
                }

                tensor = Tensor.From<float>(floats, shape);
            }

            values.Add(inputVar, Value.FromTensor(tensor));

            return (IReadOnlyDictionary<Var, IValue>)values;
        }).ToAsyncEnumerable();
    }

    public int? Count { get; }

    public IAsyncEnumerable<IReadOnlyDictionary<Var, IValue>> Samples { get; }

    /// <summary>
    /// Loads an image, letterboxes it to the model input size if needed (gray 114 padding),
    /// and converts it to planar RGB bytes (NCHW, 0..255).
    /// </summary>
    private static byte[] LoadImageToNchwUInt8(string imagePath, int targetWidth, int targetHeight)
    {
        using var source = SKBitmap.Decode(imagePath)
            ?? throw new InvalidOperationException($"Failed to decode image '{imagePath}'.");

        // 1. Calculate Letterbox proportions
        float scale = Math.Min((float)targetWidth / source.Width, (float)targetHeight / source.Height);
        int newWidth = Math.Max(1, (int)Math.Round(source.Width * scale));
        int newHeight = Math.Max(1, (int)Math.Round(source.Height * scale));
        int padX = (targetWidth - newWidth) / 2;
        int padY = (targetHeight - newHeight) / 2;

        // 2. Draw (and resize, if needed) onto an opaque RGBA canvas filled with 114 gray.
        var canvasInfo = new SKImageInfo(targetWidth, targetHeight, SKColorType.Rgba8888, SKAlphaType.Opaque);
        using var canvasBitmap = new SKBitmap(canvasInfo);
        using (var canvas = new SKCanvas(canvasBitmap))
        {
            canvas.Clear(new SKColor(LetterboxGray, LetterboxGray, LetterboxGray));

            // Same size: plain copy, no resampling. Otherwise: linear filter with mipmaps (antialiased downscale).
            var sampling = (newWidth == source.Width && newHeight == source.Height)
                ? new SKSamplingOptions(SKFilterMode.Nearest)
                : new SKSamplingOptions(SKFilterMode.Linear, SKMipmapMode.Linear);

            using var image = SKImage.FromBitmap(source);
            canvas.DrawImage(image, SKRect.Create(padX, padY, newWidth, newHeight), sampling);
        }

        // 3. Convert RGBA bytes to planar bytes (R plane, G plane, B plane).
        ReadOnlySpan<byte> pixels = canvasBitmap.GetPixelSpan();
        int rowBytes = canvasBitmap.RowBytes;
        int planeSize = targetHeight * targetWidth;
        byte[] nchwBuffer = new byte[3 * planeSize];
        for (int y = 0; y < targetHeight; y++)
        {
            int rowOffset = y * rowBytes;
            for (int x = 0; x < targetWidth; x++)
            {
                int p = rowOffset + (x * 4);
                int pixelIndex = (y * targetWidth) + x;
                nchwBuffer[pixelIndex] = pixels[p]; // Red plane
                nchwBuffer[planeSize + pixelIndex] = pixels[p + 1]; // Green plane
                nchwBuffer[(2 * planeSize) + pixelIndex] = pixels[p + 2]; // Blue plane
            }
        }

        return nchwBuffer;
    }
}
