package com.siddu.offlinedocai.data

import androidx.room.TypeConverter
import java.nio.ByteBuffer

class FloatArrayConverter {
    gitcompanion object {
        @TypeConverter
        @JvmStatic
        fun fromFloatArray(value: FloatArray?): ByteArray? {
            if (value == null) return null
            val byteBuffer = ByteBuffer.allocate(value.size * 4)
            value.forEach { byteBuffer.putFloat(it) }
            return byteBuffer.array()
        }

        @TypeConverter
        @JvmStatic
        fun toFloatArray(bytes: ByteArray?): FloatArray? {
            if (bytes == null) return null
            val byteBuffer = ByteBuffer.wrap(bytes)
            val floatArray = FloatArray(bytes.size / 4)
            for (i in floatArray.indices) {
                floatArray[i] = byteBuffer.getFloat()
            }
            return floatArray
        }
    }
}
