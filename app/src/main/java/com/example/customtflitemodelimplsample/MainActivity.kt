package com.example.customtflitemodelimplsample

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.os.Bundle
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder


class MainActivity : AppCompatActivity() {

    private var interpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        createInterpreter()
        val bitmap = loadBitmapFromAsset()
        findViewById<ImageView>(R.id.originalImage).setImageBitmap(bitmap)
        val inferenceResult = runInference(bitmap)
        val image = convertOutputArrayToImage(inferenceResult)
        saveImage(image)
    }

    private fun saveImage(finalBitmap: Bitmap) {
        findViewById<ImageView>(R.id.cartoonImage).setImageBitmap(finalBitmap)
    }

    private fun convertOutputArrayToImage(inferenceResult: Array<Array<Array<FloatArray>>>): Bitmap {
        val output = inferenceResult[0]
        val bitmap = Bitmap.createBitmap(224, 224, Bitmap.Config.ARGB_8888)
        val pixels = IntArray(224 * 224)

        var index = 0

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val b = (output[y][x][0] + 1) * 127.5
                val r = (output[y][x][1] + 1) * 127.5
                val g = (output[y][x][2] + 1) * 127.5

                val a = 0xFF
                pixels[index] = a shl 24 or (r.toInt() shl 16) or (g.toInt() shl 8) or b.toInt()
                index++
            }
        }
        bitmap.setPixels(pixels, 0, 224, 0, 0, 224, 224)
        return bitmap
    }

    private fun createInterpreter() {
        val tfLiteOptions = Interpreter.Options()
        interpreter = getInterpreter(this, TFLITE_MODEL_NAME, tfLiteOptions)
    }

    private fun loadBitmapFromAsset(): Bitmap {
        val inputStream: InputStream = applicationContext.assets.open("test-image.jpeg")
        val bitmap = BitmapFactory.decodeStream(inputStream)
        return Bitmap.createScaledBitmap(bitmap, 224, 224, false)
    }

    private fun runInference(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val outputArr = Array(1) {
            Array(224) {
                Array(224) {
                    FloatArray(3)
                }
            }
        }
        val byteBuffer = convertBitmapToByteBuffer(bitmap, 224, 224)
        interpreter?.run(byteBuffer, outputArr)
        return outputArr
    }

    private fun getInputImage(width: Int, height: Int): ByteBuffer {
        val inputImage =
            ByteBuffer.allocateDirect(1 * width * height * 3 * 4)// input image will be required input shape of tflite model
        inputImage.order(ByteOrder.nativeOrder())
        inputImage.rewind()
        return inputImage
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        // these value can be different for each channel if they are not then you may have single value instead of an array
        val mean = arrayOf(127.5f, 127.5f, 127.5f)
        val standard = arrayOf(127.5f, 127.5f, 127.5f)

        val inputImage = getInputImage(width, height)
        val intValues = IntArray(width * height)
        bitmap.getPixels(intValues, 0, width, 0, 0, width, height)

        for (y in 0 until width) {
            for (x in 0 until height) {
                val px = bitmap.getPixel(x, y)

                // Get channel values from the pixel value.
                val r = Color.red(px)
                val g = Color.green(px)
                val b = Color.blue(px)

                // Normalize channel values to [-1.0, 1.0]. This requirement depends on the model.
                // For example, some models might require values to be normalized to the range
                // [0.0, 1.0] instead.
                val rf = (r - mean[0]) / standard[0]
                val gf = (g - mean[0]) / standard[0]
                val bf = (b - mean[0]) / standard[0]

                inputImage.putFloat(bf)
                inputImage.putFloat(rf)
                inputImage.putFloat(gf)
            }
        }
        return inputImage
    }

    private fun getInterpreter(
        context: Context,
        modelName: String,
        tfLiteOptions: Interpreter.Options
    ): Interpreter {
        return Interpreter(FileUtil.loadMappedFile(context, modelName), tfLiteOptions)
    }

    companion object {
        const val TFLITE_MODEL_NAME = "lite_model_cartoongan.tflite"
    }
}