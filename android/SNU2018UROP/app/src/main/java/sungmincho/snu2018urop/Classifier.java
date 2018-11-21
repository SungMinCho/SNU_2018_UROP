package sungmincho.snu2018urop;

import android.graphics.Bitmap;

public interface Classifier {
    String classifyFrame(Bitmap bitmap);
    void close();
}
