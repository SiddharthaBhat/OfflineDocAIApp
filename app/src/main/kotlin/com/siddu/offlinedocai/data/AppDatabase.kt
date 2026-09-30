package com.siddu.offlinedocai.data

import androidx.room.Database
import androidx.room.RoomDatabase
import androidx.room.TypeConverters
import androidx.room.migration.Migration
import androidx.sqlite.db.SupportSQLiteDatabase
import com.siddu.offlinedocai.data.dao.DocumentDao
import com.siddu.offlinedocai.data.dao.ChunkDao
import com.siddu.offlinedocai.data.dao.ConversationDao
import com.siddu.offlinedocai.data.dao.MessageDao
import com.siddu.offlinedocai.data.entities.DocumentEntity
import com.siddu.offlinedocai.data.entities.ChunkEntity
import com.siddu.offlinedocai.data.entities.ConversationEntity
import com.siddu.offlinedocai.data.entities.MessageEntity

@Database(entities = [DocumentEntity::class, ChunkEntity::class, ConversationEntity::class, MessageEntity::class], version = 3, exportSchema = false)
@TypeConverters(Converters::class, FloatArrayConverter::class)
abstract class AppDatabase : RoomDatabase() {
    abstract fun documentDao(): DocumentDao
    abstract fun chunkDao(): ChunkDao
    abstract fun conversationDao(): ConversationDao
    abstract fun messageDao(): MessageDao

    companion object {
        val MIGRATION_2_3 = object : Migration(2, 3) {
            override fun migrate(db: SupportSQLiteDatabase) {
                db.execSQL("ALTER TABLE messages ADD COLUMN thinkingDurationMs INTEGER DEFAULT NULL")
            }
        }
    }
}
