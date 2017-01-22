// mcs /r:System.Drawing.dll getExamples.cs
// mono getExamples.exe

using System.IO;
using System.Drawing;
using System.Drawing.Imaging;

public class getExamples {
  const int SIZE = 28 * 28;
  const int SAMPLES = 1000;

  static public void Main () {
      ReadData(9);

  }

  static private void ReadData(int n) {
    byte[] bytes = new byte[SIZE * SAMPLES];
    FileStream fs = File.OpenRead("data"+n);
    fs.Read(bytes, 0, SIZE * SAMPLES);
    fs.Close();

    for (int i = 0; i < SAMPLES; i++) {
        Bitmap temp = new Bitmap(28, 28);
        for (int j = 0; j < 28; j++) {
          for (int k = 0; k < 28; k++) {
            byte yo = (byte)(0xff - bytes[i * SIZE + 28 * k + j]);
            temp.SetPixel(j, k, Color.FromArgb(yo, yo, yo));
          }
        }
        temp.Save(n+"_" + i + ".jpg", ImageFormat.Jpeg);
    }

  }

}
  