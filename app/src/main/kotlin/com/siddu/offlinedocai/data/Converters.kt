package com.siddu.offlinedocai.data

import androidx.room.TypeConverter
import java.util.Date

class Converters {
    companion object {
        @TypeConverter
        @JvmStatic
        fun fromTimestamp(value: Long?): Date? = value?.let { Date(it) }

        @TypeConverter
        @JvmStatic
        fun dateToTimestamp(date: Date?): Long? = date?.time
    }
}
