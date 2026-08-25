package com.siddu.offlinedocai.data.dao

import androidx.room.Dao
import androidx.room.Insert
import androidx.room.OnConflictStrategy
import androidx.room.Query
import androidx.room.Update
import com.siddu.offlinedocai.data.entities.MessageEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface MessageDao {
    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insert(message: MessageEntity): Long

    @Update
    suspend fun update(message: MessageEntity)

    @Query("SELECT * FROM messages WHERE conversationId = :convId ORDER BY timestamp ASC")
    fun getMessagesForConversation(convId: Long): Flow<List<MessageEntity>>

    @Query("DELETE FROM messages WHERE conversationId = :convId")
    suspend fun deleteMessagesForConversation(convId: Long)
}
